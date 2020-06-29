"""
Distributed training code from imagenet
"""
import os
import shutil
import time
import socket

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import torchvision.datasets as datasets
import torchvision.models as models

import sys
import utils
from utils import Logger, AverageMeter, save_checkpoint, adjust_learning_rate, accuracy

import pruning


best_prec1 = 0
batch_time = AverageMeter()
total_training_time = 0.0
total_validation_time = 0.0
rank = 0


def main(args):
    global best_prec1
    global rank

    rank = 0
    local_rank = 0
    device = "cuda"
    args.distributed = args.world_size > 1
    if args.distributed:
        if args.dist_url.startswith("tcp://"):
            if args.rank != -1:
                rank = args.rank
                local_rank = rank % 8
            else:
                local_rank = int(os.environ.get("SLURM_LOCALID"))
                task_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE").split('(')[0])
                rank = int(os.environ.get("SLURM_NODEID")) * \
                        task_per_node + \
                        local_rank
            print("Rank: {}, host: {}".format(rank, socket.gethostname()))
            dist.init_process_group(backend=args.dist_backend,
                                    init_method=args.dist_url,
                                    world_size=args.world_size,
                                    rank=rank)
        else:
            dist.init_process_group(backend=args.dist_backend,
                                    init_method=args.dist_url,
                                    world_size=args.world_size)
        device = 'cuda:{}'.format(local_rank)
        torch.cuda.set_device(local_rank)

    ### Create log dir
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    if rank == 0 and not os.path.exists(checkpoint_dir):
        print (checkpoint_dir)
        os.makedirs(checkpoint_dir)

    # create model
    model = utils.get_model(args, device, local_rank)

    # Load mask
    model_mask = None
    if args.load_mask is not None:
        mask_dict = torch.load(args.load_mask, map_location=device)
        model_mask = mask_dict["{}_{}_{}_{}".format(args.mask_type, args.mask_ratio,
                                                    args.mask_reg, args.mask_pi)]
        print("=> loaded mask '{} ({}_{}_{}_{})'".format(args.load_mask, args.mask_type, args.mask_ratio,
                                                         args.mask_reg, args.mask_pi))
        if args.load_weight is not None:
            utils.load_weights(args.load_weight, model, None, dataparallel_prefix=(not args.distributed), device=device)
            #print("=> loaded weight '{}'".format(args.load_weight))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            utils.load_weights(args.resume, model, optimizer, dataparallel_prefix=(not args.distributed), device=device)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # data loading code
    train_loader, val_loader, train_sampler = utils.get_dataloader(args)

    if args.evaluate:
        validate(val_loader, model, model_mask, criterion, device, args)
        return

    ### Save model after initialization
    args.start_epoch = 0
    if rank == 0:
        #resume = not (args.resume == '')
        if args.load_mask is not None:
            resume = False
        logger = Logger(os.path.join(checkpoint_dir, args.name),
                        ("epoch", "lr", "train time", "train loss", "train accuracy",
                         "validation loss", "validation accuracy"), resume)
        if not resume:
            checkpointfile = os.path.join(checkpoint_dir, args.name + '_chkpt_{}.pth.tar'.format(-1))
            save_checkpoint({
                'epoch': 0,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': -1,
                'optimizer' : optimizer.state_dict(),
            }, False, filename=checkpointfile)


    if args.load_mask is not None:
        if rank == 0:
            print("Before Pruning")
        #train_bef_pru = validate(train_loader, model, None, criterion, device, args)
        val_bef_pru = validate(val_loader, model, None, criterion, device, args)
        train_bef_pru = val_bef_pru
        if rank == 0:
            print("After Pruning")
       # train_aft_pru = validate(train_loader, model, model_mask, criterion, device, args)
        val_aft_pru = validate(val_loader, model, model_mask, criterion, device, args)
        train_aft_pru = val_aft_pru
        if rank == 0:
            print ("Train Bef/Aft", train_bef_pru, train_aft_pru)
            print ("Val Bef/Aft", val_bef_pru, val_aft_pru)
            logger.log((-2, 0, 0, train_bef_pru[0], train_bef_pru[1], val_bef_pru[0], val_bef_pru[1]))
            logger.log((-1, 0, 0, train_aft_pru[0], train_aft_pru[1], val_aft_pru[0], val_aft_pru[1]))
        return

    #import pdb; pdb.set_trace()
    training_start_time = time.time()
    train_err = []
    test_err = []


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        t0 = time.time()
        train_loss, train_prec1 = train(train_loader, model, model_mask, criterion, optimizer, epoch, device, args)
        t1 = time.time() - t0
        train_err.append(100 - train_prec1)

        # evaluate on validation set
        valid_loss, valid_prec1 = validate(val_loader, model, model_mask, criterion, device, args)
        test_err.append(100 - valid_prec1)

        if rank == 0:
            logger.log((epoch+1, optimizer.param_groups[0]['lr'], t1, train_loss, train_prec1, valid_loss, valid_prec1))
            # remember best prec@1 and save checkpoint
            is_best = valid_prec1 > best_prec1
            best_prec1 = max(valid_prec1, best_prec1)
            if (epoch+1) % 10 == 0:
                checkpointfile = os.path.join(checkpoint_dir, args.name + '_chkpt_{}.pth.tar'.format(epoch))
            else:
                checkpointfile = os.path.join(checkpoint_dir, args.name + '_chkpt.pth.tar')
            checkpointfile_best = os.path.join(checkpoint_dir, args.name + 'best_chkpt.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=checkpointfile, best_filename=checkpointfile_best)

    training_end_time = time.time()
    print('Training takes {}'.format(training_end_time -training_start_time))
    print(train_err)
    print(test_err)


def train(train_loader, model, model_mask, criterion, optimizer, epoch, device, args):
    global batch_time
    global total_training_time
    global rank
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mask_pass_time = AverageMeter()
    forward_pass_time = AverageMeter()
    backward_pass_time = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    epoch_st = time.time()
    torch.cuda.synchronize()
    for i, (input, target) in enumerate(train_loader):
        momentum_scale = adjust_learning_rate(optimizer, epoch, len(train_loader), i, args)
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)
        # Mask model
        if model_mask is not None:
            pruning.mask_net(model, model_mask, args.mask_bias)
        mask_pass_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        forward_pass_time.update(time.time() - end)
        loss.backward()
        backward_pass_time.update(time.time() - end)
        #optimizer.step(scale=momentum_scale)
        optimizer.step()
        momentum_scale = None
        # measure elapsed time
        if i != 0:
            batch_time.update(time.time() - end)
        end = time.time()

        if rank == 0 and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'fw time {fw.val:.3f} ({fw.avg:.3f})\t'
                  'mask time {mask.val:.3f} ({mask.avg:.3f})\t'
                  'bw time {bw.val:.3f} ({bw.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5,
                      fw=forward_pass_time, mask=mask_pass_time, bw=backward_pass_time))
    if rank == 0:
        print("Epoch elapsed: %.3f seconds" % (end - epoch_st))   
        print("End time: %.3f seconds" % end)
    end = time.time()
    if model_mask is not None:
        pruning.mask_net(model, model_mask, args.mask_bias)
    total_training_time = total_training_time + (end - epoch_st)
    if rank == 0:
        print("Total training time: %.3f seconds" % total_training_time)
    return losses.avg, top1.avg


def validate(val_loader, model, model_mask, criterion, device, args):
    global total_validation_time
    global rank
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    val_st = time.time()
    #print("Start validate time: %.3f seconds" % val_st)
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        # Mask model
        if model_mask is not None:
            pruning.mask_net(model, model_mask, args.mask_bias)
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            #print("Validate elapsed: %.3f seconds" % (end - val_st))

    end = time.time()
    total_validation_time = total_validation_time + (end - val_st)
    if rank == 0: 
        print("Total validation time: %.3f seconds" % total_validation_time)
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg


if __name__ == '__main__':
    parser = utils.prepare_parser()
    args = parser.parse_args()
    main(args)


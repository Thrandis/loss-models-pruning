import os
import sys
import time
import math
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.nn.parameter import Parameter

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
mask_names = ['MP', 'LM', 'QM', 'OBD']

def prepare_parser():
  usage = 'Parser for all scripts.'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
  parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                      choices=model_names,
                      help='model architecture: ' +
                      ' | '.join(model_names) +
                      ' (default: resnet18)')
  parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                      help='number of data loading workers (default: 4)')
  parser.add_argument('--epochs', default=90, type=int, metavar='N',
                      help='number of total epochs to run')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                      help='manual epoch number (useful on restarts)')
  parser.add_argument('-b', '--batch-size', default=256, type=int,
                      metavar='N', help='mini-batch size (default: 256)')
  parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                      metavar='LR', help='initial learning rate')
  parser.add_argument('--lr_decay', '--learning-rate_deacy', default=1, type=int,
                      metavar='LR_decay', help='Activate lr decay')
  parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                      help='momentum')
  parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                      metavar='W', help='weight decay (default: 1e-4)')
  parser.add_argument('--print-freq', '-p', default=10, type=int,
                      metavar='N', help='print frequency (default: 10)')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='path to latest checkpoint (default: none)')
  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                      help='evaluate model on validation set')
  parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                      help='use pre-trained model')
  parser.add_argument('--world-size', default=1, type=int,
                      help='number of distributed processes')
  parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                      help='url used to set up distributed training')
  parser.add_argument('--dist-backend', default='gloo', type=str,
                      help='distributed backend')
  parser.add_argument('--rank', default=-1, type=int,
                      help='rank of the distributed node')
  parser.add_argument('--name', default='sgd', type=str, help='filename used for logging')
  parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str, help='directory for where to store checkpoints')
  parser.add_argument('--checkpoint-name', default='resnet50best_chkpt', type=str, help='basename of the model checkpoint')
  ### Marsk arguments
  parser.add_argument('--load-mask', default=None, type=str,
                      help='path to the pruning mask')
  parser.add_argument('--save-mask', default="mask", type=str,
                      help='where to save the mask when computing it')
  parser.add_argument('--mask_type', default='MP', choices=mask_names, help='type of the pruning mask')
  parser.add_argument('--mask_ratio', default=0.9, type=float, help='pruning ratio')
  parser.add_argument('--mask_reg', default=0.0, type=float, help='Mask regularisation.')
  parser.add_argument('--mask_n', default=1600, type=int, help='Number os sample use to esimate the masks.')
  parser.add_argument('--mask_pi', default=1, type=int, help='Mask pruning iterations.')
  parser.add_argument('--mask_bias', default=False, type=float, help='is the pruning mask applied to bias?')
  parser.add_argument('--load_weight', default=None, type=str,
                      help='path of model checkpoint which will be used to initialize the weight')
  return parser


def get_model(args, device="cuda", local_rank=0):
  if args.pretrained:
     print("=> using pre-trained model '{}'".format(args.arch))
     model = models.__dict__[args.arch](pretrained=True)
  else:
     print("=> creating model '{}'".format(args.arch))
     model = models.__dict__[args.arch]()
  if not args.distributed:
     model = model.to(device)
  else:
     # Distributed training uses 4 tricsk to maintain accuracy with much larger
     # batch sizes. See https://arxiv.org/pdf/1706.02677.pdf for more details
     if args.arch.startswith('resnet'):
       for m in model.modules():
          # Trick 1: the last BatchNorm layer in each block needs to be initialized
          # as zero gamma
          if isinstance(m, BasicBlock):
             num_features = m.bn2.num_features
             m.bn2.weight = Parameter(torch.zeros(num_features))
          if isinstance(m, Bottleneck):
             num_features = m.bn3.num_features
             m.bn3.weight = Parameter(torch.zeros(num_features))
          ## Trick 2: Linear layers are initialized by drawing weights from a
          # zero-mean Gaussian with stddev 0.01. In the paper it was only for
          # fc layer, but in practice this seems to give better accuracy
          if isinstance(m, nn.Linear):
             m.weight.data.normal_(0, 0.01)

     model.to(device)
     #device = args.rank % args.world_size
     print("Using device: {}".format(device))
     model = torch.nn.parallel.DistributedDataParallel(model,
                                                       device_ids=[local_rank],
                                                       output_device=local_rank)
  return model


def get_mask(args, device="cuda"):
  mask = None
  mask_dict = torch.load(args.load_mask, map_location=device)
  model_mask = mask_dict["{}_{}_{}_{}".format(args.mask_type, args.mask_ratio, args.mask_reg, args.mask_pi)]
  print("=> loaded mask '{} ({}_{})'".format(args.load_mask, args.mask_type, args.mask_ratio))
  return model_mask

def get_dataloader(args): 
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    return train_loader, val_loader, train_sampler

def save_checkpoint(state, is_best, filename='sgd_checkpoint.pth.tar',
                    best_filename='best_sgd_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def load_weights(model_file, model, optim=None,  strict=True, device="cuda", dataparallel_prefix=False):
   checkpoint = torch.load(model_file, map_location=device)
   start_epoch = checkpoint['epoch']
   best_prec1 = checkpoint['best_prec1']
   model_dict = checkpoint['state_dict']
   if dataparallel_prefix:
     model_dict = dict((key[7:], value) for key, value in model_dict.items())
   model.load_state_dict(model_dict, strict=strict)
   if optim is not None:
     optim.load_state_dict(checkpoint['optimizer'])
   print("=> loaded checkpoint '{}' (epoch {}, acc {})".format(model_file, checkpoint['epoch'], best_prec1))


def adjust_learning_rate(optimizer, epoch, num_iter_in_one_epoch, iter_index, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.distributed:
        # Trick 3: lr scales linearly with world size with warmup
        # num_iter = num_iter_in_one_epoch
        ref_lr = 0.1 * args.world_size * args.batch_size / 256
        if args.lr_decay == 0:
          lr = ref_lr
        elif epoch < 5:
            # lr_step = (args.world_size - 1) * args.lr / (5.0 * num_iter)
            # lr = args.lr + (epoch * num_iter + iter_index) * lr_step
            base_lr = ref_lr / args.world_size
            lr_step = (ref_lr - base_lr) / (5 * num_iter_in_one_epoch)
            lr = base_lr + (epoch * num_iter_in_one_epoch + iter_index) * lr_step
        elif epoch < 80:
            # lr = args.world_size * args.lr * (0.1 ** (epoch // 30))
            lr = ref_lr * (0.1 ** (epoch // 30))
        else:
            # lr = args.world_size * args.lr * (0.1 ** 3)
            lr = ref_lr * (0.1 ** 3)
        for param_group in optimizer.param_groups:
            lr_old = param_group['lr']
            param_group['lr'] = lr
            #print("lr={}".format(lr))
            # Trick 4: apply momentum correction when lr is updated
            if lr > lr_old:
                param_group['momentum'] = lr / lr_old * args.momentum
            else:
                param_group['momentum'] = args.momentum
    else:
        if args.lr_decay == 0 :
          lr = args.lr
        else:
          lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ProgressBar:
	def __init__(self):
		_, term_width = os.popen('stty size', 'r').read().split()
		self.term_width = int(term_width)

		self.TOTAL_BAR_LENGTH = int(self.term_width / 2)
		self.last_time = time.time()
		self.begin_time = self.last_time

	def update(self, current, total, msg=None):
	    #global last_time, begin_time
	    if current == 0:
	        self.begin_time = time.time()  # Reset for new bar.

	    cur_len = int(self.TOTAL_BAR_LENGTH*current/total)
	    rest_len = int(self.TOTAL_BAR_LENGTH - cur_len) - 1

	    sys.stdout.write(' [')
	    for i in range(cur_len):
	        sys.stdout.write('=')
	    sys.stdout.write('>')
	    for i in range(rest_len):
	        sys.stdout.write('.')
	    sys.stdout.write(']')

	    cur_time = time.time()
	    step_time = cur_time - self.last_time
	    self.last_time = cur_time
	    tot_time = cur_time - self.begin_time

	    L = []
	    L.append('  Step: %s' % format_time(step_time))
	    L.append(' | Tot: %s' % format_time(tot_time))
	    if msg:
	        L.append(' | ' + msg)

	    msg = ''.join(L)
	    sys.stdout.write(msg)
	    for i in range(self.term_width-int(self.TOTAL_BAR_LENGTH)-len(msg)-3):
	        sys.stdout.write(' ')

	    # Go back to the center of the bar.
	    for i in range(self.term_width-int(self.TOTAL_BAR_LENGTH/2)+2):
	        sys.stdout.write('\b')
	    sys.stdout.write(' %d/%d ' % (current+1, total))

	    if current < total-1:
	        sys.stdout.write('\r')
	    else:
	        sys.stdout.write('\n')
	    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class Logger:
    """Simple class for logging

    Args:
        name (string): suffix-free filename (".log" will be appended automatically)
        params (tuple): tuple of strings of names of the parameters to be logged (column headers)

    """
    def __init__(self, name, params, resume=False):
        self.filename = name + ".log"
        self.n_params = len(params)

        if not resume:
          with open(self.filename, 'a') as f:
            for idx, p in enumerate(params):
              if idx > 0:
                f.write(',')
              f.write(p)
            f.write('\n')



    """Log parameter values

    Args:
        values (tuple): tuple of values to be logged

    Note: There should be one value for each parameter listed when the Logger was constructed
    """
    def log(self, values):
        assert len(values) == self.n_params, "length of values doesn't match the number of parameter names passed when this Logger was created"
        with open(self.filename, 'a') as f:
            for idx, p in enumerate(values):
                if idx > 0:
                    f.write(',')
                if type(p) is str:
                    f.write(p)
                else:
                    f.write(str(p))
            f.write('\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Wrap Pruning functionin the model class,
#tentative to speed-up prunning when applied to a DistributedDataParallel Model
class PrunningWrapper(nn.Module):
  def __init__(self, model, mask, bias=False):
    self.model = model
    self.mask = mask
    self.bias = bias

  def forward(self, x):
    ### Apply Mask
    # if self.bias:
    #   for n, p in net.named_parameters():
    #     p.data.masked_fill_((m.eq(0)).view(p.shape).to(p.device), 0)
    # for n, p in net.named_parameters() if 'bias' not in n
    # p for n, p in net.named_parameters()

    ## Apply model forward
    return model.forward(x)
    

import copy
import os
import sys
import subprocess
import argparse
import time
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

import log
from models import MLP_300_100, VGG11, PreActResNet18
from utils import (train_loop, eval_loop, kaiming_init, get_cifar10_loader,
                   get_cifar10_loader_gpu, get_mnist_sets)
from pruning import get_mask_iteratively, mask_net


DATA_PATH = os.path.join(os.environ['SLURM_TMPDIR'], 'data')
XP_PATH = os.path.join(os.environ['SLURM_TMPDIR'], 'results')


def get_model(args):
    # Build network
    log.log_comment('Building Model')
    if args.arch == 'MLP':
        net = MLP_300_100()
    elif args.arch == 'VGG11':
        net = VGG11()
    elif args.arch == 'PreActResNet18':
        net = PreActResNet18()
    else:
        raise NotImplementedError
    net.to('cuda')
    # Initialize it
    kaiming_init(net)
    # Print it
    log.log_comment('Network:')
    for s in str(net).split('\n'):
        log.log_comment(s)
    n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.log_comment('  - Number of parameters: %i' % n)
    return net


def main():
    # -------------------------------------------------------------------------
    # Argumet Parser

    parser = argparse.ArgumentParser(description='SNIP on CIFAR10')

    # Lottery Ticket
    parser.add_argument('--pr', type=float, default=0.956,
                        help='Pruning ratio')
    parser.add_argument('--pm', type=str, default='MP',
                        choices=['MP', 'LM', 'QM', 'OBD'],
                        help='Pruning method')
    parser.add_argument('--pi', type=int, default=1,
                        help='Number of iterations of pruning.')
    parser.add_argument('--pe', type=int, default=0,
                        help='Iteration before which we prune.')
    parser.add_argument('--reg', type=float, default=0.0,
                        help='Regularization parameter for snip and obsn.')
    parser.add_argument('--nex', type=int, default=1000,
                        help='Number of examples for pruning algorithms')
    parser.add_argument('--exp', action='store_true',
                        help='Use eponential pruning steps')

    # Model
    parser.add_argument('--arch', default='VGG11', type=str,
                        choices=['MLP', 'VGG11', 'PreActResNet18'],
                        help='Model achitecture.')
    parser.add_argument('--model_0', type=str, default=None,
                        help='Path to a saved network.')

    # Optimizer
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate.')
    parser.add_argument('--dec', default=1., type=float,
                        help='Learning rate decay (exponential).')
    parser.add_argument('--dec_every', default=1, type=int,
                        help='Number of epochs between decay.')
    parser.add_argument('--l2', default=0.0005, type=float,
                        help='L2 regularization.')

    # Experiment Management
    parser.add_argument('--path', type=str, default=XP_PATH,
                        help='Path for experiment')
    parser.add_argument('--data_path', default=DATA_PATH, type=str,
                        help='Path for the data')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to prepare data.')
    parser.add_argument('--nepochs', default=200, type=int,
                        help='Number of epochs to run.')
    parser.add_argument('--seed', default=1111, type=int,
                        help='Seed of the random number generators.')
    args = parser.parse_args()

    assert torch.cuda.is_available()  # Only train on GPU
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -------------------------------------------------------------------------
    # Preparing Paths and Log

    name = ''
    for k, v in sorted(args.__dict__.items(), key=lambda a: a[0]):
        if k not in ['path', 'data_path', 'num_workers']:
            if k == 'model_0' and v is not None:
                v = True
            name += '%s=%s,' % (k, str(v))
    name = name[:-1]

    if args.path is not None:
        xp_path = os.path.join(args.path, name)
        if not os.path.isdir(xp_path):
            os.makedirs(xp_path)
        else:
            sys.exit('Experiment already exists!')
        log_path = os.path.join(xp_path, 'log.txt')
    else:
        log_path = None
    log.prepare_log(log_path)
    log.log_comment('Pruning Experiment')
    try:
        repo = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       encoding='utf-8').strip()
    except:
        repo = 'None'
    log.log_comment('Git commit: ' + repo)
    log.log_comment('Arguments:')
    for k in sorted(args.__dict__.keys()):
        log.log_comment('  - %s %s' % (k, str(args.__dict__[k])))

    # -------------------------------------------------------------------------
    # Preparing Data Streams

    log.log_comment('Preparing data')
    if args.arch == 'MLP':
        train_set, valid_set, test_set = get_mnist_sets(args.seed,
                                                        args.data_path)
        train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
        valid_loader = (valid_set.tensors,)
        test_loader = (test_set.tensors,)
        fisher_loader = DataLoader(train_set, batch_size=args.nex,
                                   shuffle=True)
    else:
        train_loader = get_cifar10_loader('train', 100, args.num_workers, True,
                                          args.data_path)
        valid_loader = get_cifar10_loader_gpu('valid', 5000, args.num_workers,
                                              False, args.data_path)
        test_loader = get_cifar10_loader_gpu('test', 5000, args.num_workers,
                                              False, args.data_path)
        fisher_loader = get_cifar10_loader_gpu('train', 100, args.num_workers,
                                               True, args.data_path,
                                               which_transform='valid')

    # -------------------------------------------------------------------------
    # Preparing Model and Optimizer

    if args.model_0 is not None:
        net = torch.load(args.model_0)
    else:
        net = get_model(args)
    mask = None
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.dec_every,
                                                gamma=args.dec,
                                                last_epoch=-1)

    # -------------------------------------------------------------------------
    # Main Loop

    form = ['%i', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.2f']
    head = ['n', 'train_l', 'train_m', 'valid_l', 'valid_m', 'test_l',
            'test_m', 'time']
    log.log_head(head, form)
    best_err = float('inf')
    best_net = copy.deepcopy(net)

    for epoch in range(0, args.nepochs):
        torch.manual_seed(args.seed + epoch)
        torch.cuda.manual_seed_all(args.seed + epoch)
        random.seed(args.seed + epoch)

        if epoch == args.pe:
            net = best_net

            log.log_comment('Performances before pruning')
            timer = time.time()
            train_l, train_m = eval_loop(net, fisher_loader)
            valid_l, valid_m = eval_loop(net, valid_loader)
            test_l, test_m = eval_loop(net, test_loader)
            log.log_comment('\t'.join(form) % (epoch, train_l, train_m,
                                               valid_l, valid_m, test_l,
                                               test_m, time.time() - timer))

            log.log_comment('Pruning network!')
            timer = time.time()

            # Compute pruning ratios
            if args.exp:

                def get_prs(pr, pi, p0=0):
                    r = 1 - (1 - (pr - p0)) ** (1 / pi)
                    p = [0]
                    for i in range(pi):
                        p.append(p[-1] + (1 - p[-1]) * r)
                    return [p0 + pp for pp in p[1:]]

                prunings = get_prs(args.pr, args.pi)
                log.log_comment('Final pruning: ' + str(prunings[-1]))
            else:
                prunings = [args.pr / args.pi * (i + 1) for i in range(args.pi)] 

            mask = get_mask_iteratively(args.pm, net, prunings,
                                        loader=fisher_loader, n=args.nex,
                                        reg=args.reg)
            mask_net(net, mask, False)

            log.log_comment('Performances after pruning')
            train_l, train_m = eval_loop(net, fisher_loader)
            valid_l, valid_m = eval_loop(net, valid_loader)
            test_l, test_m = eval_loop(net, test_loader)
            log.log_comment('\t'.join(form) % (epoch, train_l, train_m,
                                               valid_l, valid_m, test_l,
                                               test_m, time.time() - timer))

            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.l2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=args.dec_every,
                                                        gamma=args.dec,
                                                        last_epoch=-1)

        # Actual training and validation loops
        timer = time.time()
        train_l, train_m = train_loop(net, optimizer, train_loader, mask)
        valid_l, valid_m = eval_loop(net, valid_loader)
        test_l, test_m = eval_loop(net, test_loader)

        to_log = [epoch, train_l, train_m, valid_l, valid_m, test_l, test_m,
                  time.time() - timer]
        log.log_values(to_log)

        # Early stopping if not working properly
        if np.isnan(train_l):
            sys.exit()

        # Learning Rate Decay
        scheduler.step()

        # Saving net
        if valid_m < best_err:
            log.log_comment('Best model so far.')
            best_err = valid_m
            net.zero_grad()
            best_net = copy.deepcopy(net)
            torch.save(net, os.path.join(xp_path, 'best_model.pt'))
    net.zero_grad()
    torch.save(net, os.path.join(xp_path, 'last_model.pt'))


if __name__ == '__main__':
    main()

import argparse
import os
import time

import torch


import utils
import pruning
from pruning import get_mask_iteratively


def main(args):
    ## We don't support distributed pruning for now
    args.distributed = False
    which_models = [-1] #, 0, 5, 10, 20, 50, 89]
    prune = [args.mask_ratio,]
    methods = [args.mask_type]#, 'QM', 'OBD']
   # methods = ['OBD']
    nexamples = args.mask_n #1600
    reg = 0.0
    pi = args.mask_pi
    exp=True

    device = "cuda"

    # Init model
    model = utils.get_model(args, device)
    # Load mask
    #import pdb; pdb.set_trace()
    if args.load_mask is not None:
        model_mask = utils.get_mask(args, device)
    # Load data
    train_loader, valid_loader, _ = utils.get_dataloader(args)

    

    for epoch in which_models:
        masks ={}
        #model_file = os.path.join(args.checkpoint_dir, "{}{}.pth.tar".format(args.checkpoint_name, epoch))
        model_file = os.path.join(args.checkpoint_dir, "{}.pth.tar".format(args.checkpoint_name))
        if os.path.isfile(model_file):
            ## Load model weight
            print("=> loading checkpoint '{}'".format(model_file))
            utils.load_weights(model_file, model, None, dataparallel_prefix=True, device=device)
        else:
            print("=> no checkpoint found at '{}'".format(model_file))
        for method in methods:
            model_mask = None
            for level in prune:
                # Mask model if needed
                if model_mask is not None:
                    pruning.mask_net(model, model_mask, args.mask_bias)
                # Create the masks
                s1 = time.time()

                if exp:
                    def get_prs(pr, pi, p0=0):
                        r = 1 - (1 - (pr - p0)) ** (1 / pi)
                        p = [0]
                        for i in range(pi):
                            p.append(p[-1] + (1 - p[-1]) * r)
                        return [p0 + pp for pp in p[1:]]
                    prunings = get_prs(level, pi)
                else:
                    prunings = [level / pi * (i + 1) for i in range(pi)] 

                #import pdb; pdb.set_trace()
                mask = get_mask_iteratively(method, model, prunings,
                                            loader=train_loader, n=nexamples,
                                            reg=reg)

                print("Compute time {}_{}:{}".format(method, level, time.time() - s1))
                ### Save Make
                masks["{}_{}_{}_{}".format(method, level, reg, pi)] =  mask
            torch.save(masks,
                       args.save_mask + "_{}_{}.pth.tar".format(args.checkpoint_name, epoch))


if __name__ == '__main__':
    parser = utils.prepare_parser()
    args = parser.parse_args()
    main(args)

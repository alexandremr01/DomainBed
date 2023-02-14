# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import pickle

import PIL
import torch
import torchvision
import torch.utils.data
from torch import nn, optim, autograd
import numpy as np

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed import partitioners
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import domainbed.partitioners

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Domain partition')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--partitioner', type=str, default="EIIL")
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # if args.hparams_seed == 0:
    #     hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    # else:
    #     hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
    #         misc.seed_hash(args.hparams_seed, args.trial_seed))
    # if args.hparams:
    #     hparams.update(json.loads(args.hparams))
    hparams = { }

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, [ ], hparams)
    else:
        raise NotImplementedError

    partitioner_class = partitioners.get_partitioner_class(args.partitioner)
    partitioner = partitioner_class(dataset.input_shape, dataset.num_classes,
        len(dataset), hparams)

    envs = partitioner.split(dataset)
    with open(os.path.join(args.output_dir, 'partition.json'), 'w') as f:
      json.dump(envs.get_mapping_original_to_new(), f)
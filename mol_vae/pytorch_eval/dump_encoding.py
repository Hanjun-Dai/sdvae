#!/usr/bin/env python

from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random

sys.path.append('%s/../mol_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

sys.path.append('%s/../pytorch_eval' % os.path.dirname(os.path.realpath(__file__)))
from att_model_proxy import AttMolProxy

import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for encoding')
cmd_opt.add_argument('-mode', default='gpu', help='cpu/gpu')
cmd_opt.add_argument('-total', type=int, help='total number of smiles')
cmd_opt.add_argument('-round', type=int, help='encoding round')
cmd_opt.add_argument('-noisy', type=int, help='use noisy encoding')
args, _ = cmd_opt.parse_known_args()

BATCH = 10000
from tqdm import tqdm
import cPickle as cp

if __name__ == '__main__':
    cur_batch_smiles = []
    print(cmd_args)
    print(args)

    with open(cmd_args.smiles_file) as f:
        smiles_list = f.readlines()    

    if hasattr(args, 'total'):
        smiles_list = smiles_list[0:args.total]

    model = AttMolProxy()
    BATCH = cmd_args.batch_size
    latent_points_batches = []

    cfg_dump_file = '.'.join(cmd_args.smiles_file.split('.')[0:-1]) + '.cfg_dump'
    with open(cfg_dump_file, 'rb') as f:
        for i in tqdm(range(0, len(smiles_list), BATCH)):

            if i + BATCH < len(smiles_list):
                size = BATCH
            else:
                size = len(smiles_list) - i

            cfg_tree_list = []
            for j in range(i, i + size):
                n = cp.load(f)
                cfg_tree_list.append(n)

            latent_points_batches.append(model.encode(cfg_tree_list, use_random=args.noisy))
        
    latent_points = np.vstack(latent_points_batches)

    np.save('%s/noisy-%d-round-%d-features.npy' % (cmd_args.save_dir, args.noisy, args.round), latent_points)

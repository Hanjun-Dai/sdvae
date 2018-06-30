#!/usr/bin/env python

from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random

sys.path.append('%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

from att_model_proxy import AttProgProxy

import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for data dump')
cmd_opt.add_argument('-min_len', type=int, help='min # of statements')
cmd_opt.add_argument('-max_len', type=int, help='max # of statements')
cmd_opt.add_argument('-phase', type=str, help='train / test')
cmd_opt.add_argument('-data_dir', type=str, help='data folder')
cmd_opt.add_argument('-prefix', type=str, help='data prefix')

args, _ = cmd_opt.parse_known_args()

BATCH = 10000
from tqdm import tqdm
import cPickle as cp

if __name__ == '__main__':
    cur_batch_smiles = []
    print(cmd_args)
    print(args)

    program_list = []
    for l in range(args.min_len, args.max_len + 1):
        if args.phase == 'train':
            fname = '%s/%s-number-50000-nbstat-%d.txt' % (args.data_dir, args.prefix, l)
        else:
            fname = '%s/%s-number-50000-nbstat-%d.test.txt' % (args.data_dir, args.prefix, l)

        with open(fname, 'r') as f:
            for row in tqdm(f):
                program = row.strip()
                program_list.append(program)

    model = AttProgProxy()
    BATCH = cmd_args.batch_size
    latent_points_batches = []
    print('num_samples: ', len(program_list))
    for i in tqdm(range(0, len(program_list), BATCH)):

        if i + BATCH < len(program_list):
            size = BATCH
        else:
            size = len(program_list) - i

        latent_points_batches.append(model.encode(program_list[i : i + size], use_random=False))
    
    latent_points = np.vstack(latent_points_batches)

    np.save('%s/%s-%d-%d-features.npy' % (cmd_args.save_dir, args.phase, args.min_len, args.max_len), latent_points)

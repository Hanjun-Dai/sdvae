#!/usr/bin/env python

from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random

from tqdm import tqdm

sys.path.append('%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)))
from prog_util import DECISION_DIM
from prog_tree import AnnotatedTree2ProgTree
from cmd_args import cmd_args

sys.path.append('%s/../prog_decoder' % os.path.dirname(os.path.realpath(__file__)))
from prog_tree_decoder import ProgTreeDecoder, batch_make_att_masks
from tree_walker import ProgramOnehotBuilder 

sys.path.append('%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)))
import cfg_parser as parser

from joblib import Parallel, delayed
import h5py

import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for data dump')
cmd_opt.add_argument('-min_len', type=int, help='min # of statements')
cmd_opt.add_argument('-max_len', type=int, help='max # of statements')
cmd_opt.add_argument('-phase', type=str, help='train / test')
args, _ = cmd_opt.parse_known_args()

def process_chunk(program_list):
    grammar = parser.Grammar(cmd_args.grammar_file)

    cfg_tree_list = []
    for program in program_list:
        ts = parser.parse(program, grammar)
        assert isinstance(ts, list) and len(ts) == 1

        n = AnnotatedTree2ProgTree(ts[0])
        cfg_tree_list.append(n)

    walker = ProgramOnehotBuilder()
    tree_decoder = ProgTreeDecoder()
    onehot, masks = batch_make_att_masks(cfg_tree_list, tree_decoder, walker, dtype=np.byte)

    return (onehot, masks)

def run_job(L):
    chunk_size = 5000
    
    list_binary = Parallel(n_jobs=cmd_args.data_gen_threads, verbose=50)(
        delayed(process_chunk)(L[start: start + chunk_size])
        for start in range(0, len(L), chunk_size)
    )

    all_onehot = np.zeros((len(L), cmd_args.max_decode_steps, DECISION_DIM), dtype=np.byte)
    all_masks = np.zeros((len(L), cmd_args.max_decode_steps, DECISION_DIM), dtype=np.byte)

    for start, b_pair in zip( range(0, len(L), chunk_size), list_binary ):
        all_onehot[start: start + chunk_size, :, :] = b_pair[0]
        all_masks[start: start + chunk_size, :, :] = b_pair[1]

    return all_onehot, all_masks

if __name__ == '__main__':

    onehot_list = []
    mask_list = []

    for l in range(args.min_len, args.max_len + 1):
        if args.phase == 'train':
            fname = '%s/free_var_id-check_data-number-50000-nbstat-%d.txt' % (cmd_args.save_dir, l)
        else:
            fname = '%s/free_var_id-check_data-number-50000-nbstat-%d.test.txt' % (cmd_args.save_dir, l)

        program_list = []
        with open(fname, 'r') as f:
            for row in tqdm(f):
                program = row.strip()
                program_list.append(program)

        onehot, mask = run_job(program_list)
        onehot_list.append(onehot)
        mask_list.append(mask)

    all_onehot = np.vstack(onehot_list)
    all_mask = np.vstack(mask_list)

    # shuffle training set
    idxes = range(all_onehot.shape[0])
    random.shuffle(idxes)
    idxes = np.array(idxes, dtype=np.int32)
    all_onehot = all_onehot[idxes, :, :]
    all_mask = all_mask[idxes, :, :]
    print('num samples: ', len(idxes)) 
    out_file = '%s/nbstate-%d-to-%d-%s.h5' % (cmd_args.save_dir, args.min_len, args.max_len, args.phase)
    h5f = h5py.File(out_file, 'w')
    h5f.create_dataset('x_%s' % args.phase, data=all_onehot)
    h5f.create_dataset('masks_%s' % args.phase, data=all_mask)
    h5f.close()

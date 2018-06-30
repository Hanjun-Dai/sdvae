#!/usr/bin/env python

from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random

from tqdm import tqdm

sys.path.append('%s/../mol_common' % os.path.dirname(os.path.realpath(__file__)))
from mol_util import DECISION_DIM
from mol_tree import AnnotatedTree2MolTree
from cmd_args import cmd_args

sys.path.append('%s/../mol_decoder' % os.path.dirname(os.path.realpath(__file__)))
from attribute_tree_decoder import create_tree_decoder
from mol_decoder import batch_make_att_masks
from tree_walker import OnehotBuilder 

sys.path.append('%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)))
import cfg_parser as parser

from joblib import Parallel, delayed
import h5py

def process_chunk(smiles_list):
    grammar = parser.Grammar(cmd_args.grammar_file)

    cfg_tree_list = []
    for smiles in smiles_list:
        ts = parser.parse(smiles, grammar)
        assert isinstance(ts, list) and len(ts) == 1

        n = AnnotatedTree2MolTree(ts[0])
        cfg_tree_list.append(n)

    walker = OnehotBuilder()
    tree_decoder = create_tree_decoder()
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

    f_smiles = '.'.join(cmd_args.smiles_file.split('/')[-1].split('.')[0:-1])
    out_file = '%s/%s-%d.h5' % (cmd_args.save_dir, f_smiles, cmd_args.skip_deter)
    h5f = h5py.File(out_file, 'w')
    h5f.create_dataset('x', data=all_onehot)
    h5f.create_dataset('masks', data=all_masks)
    h5f.close()

if __name__ == '__main__':

    smiles_list = []
    with open(cmd_args.smiles_file, 'r') as f:
        for row in tqdm(f):
            smiles = row.strip()
            smiles_list.append(smiles)

    run_job(smiles_list)
    



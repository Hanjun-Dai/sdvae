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
from mol_tree import AnnotatedTree2MolTree
from cmd_args import cmd_args

sys.path.append('%s/../mol_decoder' % os.path.dirname(os.path.realpath(__file__)))
from mol_decoder import batch_make_att_masks

sys.path.append('%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)))
import cfg_parser as parser

import h5py

def parse_smiles_with_cfg(smiles_file, grammar_file):
    grammar = parser.Grammar(cmd_args.grammar_file)

    cfg_tree_list = []
    with open(smiles_file, 'r') as f:
        for row in tqdm(f):
            smiles = row.strip()
            ts = parser.parse(smiles, grammar)
            assert isinstance(ts, list) and len(ts) == 1
            n = AnnotatedTree2MolTree(ts[0])
            cfg_tree_list.append(n)

    return cfg_tree_list

if __name__ == '__main__':

    cfg_tree_list = parse_smiles_with_cfg(cmd_args.smiles_file, cmd_args.grammar_file)

    all_true_binary, all_rule_masks = batch_make_att_masks(cfg_tree_list)
    
    print(all_true_binary.shape, all_rule_masks.shape)

    f_smiles = '.'.join(cmd_args.smiles_file.split('/')[-1].split('.')[0:-1])

    out_file = '%s/%s.h5' % (cmd_args.save_dir, f_smiles)    
    h5f = h5py.File(out_file, 'w')

    h5f.create_dataset('x', data=all_true_binary)
    h5f.create_dataset('masks', data=all_rule_masks)
    h5f.close()

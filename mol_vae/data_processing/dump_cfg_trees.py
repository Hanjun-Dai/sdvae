#!/usr/bin/env python

from __future__ import print_function
from past.builtins import range

import os
import sys
import numpy as np
import math
import random

import torch
from torch.autograd import Variable

from joblib import Parallel, delayed

sys.path.append('%s/../mol_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args
from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree, Node

sys.path.append('%s/../mol_vae' % os.path.dirname(os.path.realpath(__file__)))
from mol_vae import MolVAE, MolAutoEncoder

sys.path.append('%s/../mol_decoder' % os.path.dirname(os.path.realpath(__file__)))
from attribute_tree_decoder import create_tree_decoder
from mol_decoder import batch_make_att_masks
from tree_walker import OnehotBuilder, ConditionalDecoder

sys.path.append('%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)))
import cfg_parser as parser

def parse_single(smiles, grammar):
    ts = parser.parse(smiles, grammar)
    assert isinstance(ts, list) and len(ts) == 1
    n = AnnotatedTree2MolTree(ts[0])
    return n

def parse_many(chunk, grammar):
    return [parse_single(smiles, grammar) for smiles in chunk]

def parse(chunk, grammar):
    size = 100
    result_list = Parallel(n_jobs=-1)(delayed(parse_many)(chunk[i: i + size], grammar) for i in range(0, len(chunk), size))
    return [_1 for _0 in result_list for _1 in _0]

import cPickle as cp

from tqdm import tqdm

if __name__ == '__main__':
    smiles_file = cmd_args.smiles_file 
    fname = '.'.join(smiles_file.split('.')[0:-1]) + '.cfg_dump'
    fout = open(fname, 'wb')
    grammar = parser.Grammar(cmd_args.grammar_file)

    with open(smiles_file, 'r') as f:
        smiles = f.readlines()
    for i in range(len(smiles)):
        smiles[ i ] = smiles[ i ].strip()

    # cfg_tree_list = parse(smiles, grammar)
    # cp.dump(cfg_tree_list, fout, cp.HIGHEST_PROTOCOL)
    
    for i in tqdm(range(len(smiles))):
        ts = parser.parse(smiles[i], grammar)
        assert isinstance(ts, list) and len(ts) == 1
        n = AnnotatedTree2MolTree(ts[0])
        cp.dump(n, fout, cp.HIGHEST_PROTOCOL)

    fout.close()

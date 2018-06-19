#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import csv
import numpy as np
import math
import random
from collections import defaultdict
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append( '%s/../mol_common' % os.path.dirname(os.path.realpath(__file__)) )
from mol_util import TOTAL_NUM_RULES, DECISION_DIM, rule_ranges, terminal_idxes
from mol_tree import Node, get_smiles_from_tree, AnnotatedTree2MolTree, AnnotatedTree2Onehot
from cmd_args import cmd_args
from pytorch_initializer import weights_init

sys.path.append( '%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)) )
import cfg_parser as parser

class CNNEncoder(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.conv1 = nn.Conv1d(DECISION_DIM, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)

        self.last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10, 435)
        self.mean_w = nn.Linear(435, latent_dim)
        self.log_var_w = nn.Linear(435, latent_dim)
        weights_init(self)

    def forward(self, x_cpu):
        if cmd_args.mode == 'cpu':
            batch_input = Variable(torch.from_numpy(x_cpu))
        else:
            batch_input = Variable(torch.from_numpy(x_cpu).cuda())

        h1 = self.conv1(batch_input)
        h1 = F.relu(h1)        
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)

        # h3 = torch.transpose(h3, 1, 2).contiguous()
        flatten = h3.view(x_cpu.shape[0], -1)
        h = self.w1(flatten)
        h = F.relu(h)

        z_mean = self.mean_w(h)
        z_log_var = self.log_var_w(h)
        
        return (z_mean, z_log_var)

if __name__ == '__main__':

    smiles_list = ['N\SCPP#IOS', 'CP\P', 'PINI']

    cfg_trees = []
    cfg_onehots = []
    grammar = parser.Grammar(cmd_args.grammar_file)
    for smiles in smiles_list:
        ts = parser.parse(smiles, grammar)
        assert isinstance(ts, list) and len(ts) == 1
        n = AnnotatedTree2MolTree(ts[0])
        cfg_trees.append(n)
        cfg_onehots.append(AnnotatedTree2Onehot(ts[0], 50))

    cfg_onehots = np.stack(cfg_onehots, axis=0)

    encoder = CNNEncoder(max_len=50, latent_dim=64)
    if cmd_args.mode == 'gpu':
        encoder.cuda()
    z = encoder(cfg_onehots)
    print(z[0].size())

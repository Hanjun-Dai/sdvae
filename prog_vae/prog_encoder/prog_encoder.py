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

sys.path.append( '%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)) )
from prog_util import DECISION_DIM
from cmd_args import cmd_args
from pytorch_initializer import weights_init

sys.path.append( '%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)) )
import cfg_parser as parser

class CNNEncoder(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.conv1 = nn.Conv1d(DECISION_DIM, cmd_args.c1, cmd_args.c1)
        self.conv2 = nn.Conv1d(cmd_args.c1, cmd_args.c2, cmd_args.c2)
        self.conv3 = nn.Conv1d(cmd_args.c2, cmd_args.c3, cmd_args.c3)

        self.last_conv_size = max_len - cmd_args.c1 + 1 - cmd_args.c2 + 1 - cmd_args.c3 + 1
        self.w1 = nn.Linear(self.last_conv_size * cmd_args.c3, cmd_args.dense)
        self.mean_w = nn.Linear(cmd_args.dense, latent_dim)
        self.log_var_w = nn.Linear(cmd_args.dense, latent_dim)
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
    pass

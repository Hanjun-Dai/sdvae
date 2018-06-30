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
from prog_util import prod, DECISION_DIM
from cmd_args import cmd_args
from pytorch_initializer import weights_init

sys.path.append( '%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)) )
import cfg_parser

from custom_loss import my_perp_loss, my_binary_loss

from tqdm import tqdm

class StateDecoder(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(StateDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.z_to_latent = nn.Linear(self.latent_dim, self.latent_dim)
        if cmd_args.rnn_type == 'gru':
            self.gru = nn.GRU(self.latent_dim, cmd_args.hidden, 3)
        else:
            raise NotImplementedError

        self.decoded_logits = nn.Linear(cmd_args.hidden, DECISION_DIM)
        weights_init(self)

    def forward(self, z):
        assert len(z.size()) == 2 # assert the input is a matrix

        h = self.z_to_latent(z)
        h = F.relu(h)

        rep_h = h.expand(self.max_len, z.size()[0], z.size()[1]) # repeat along time steps

        out, _ = self.gru(rep_h) # run multi-layer gru

        logits = self.decoded_logits(out)

        return logits

class ConditionalStateDecoder(nn.Module):
    def __init__(self, max_len, latent_dim):
        super(StateDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.z_to_latent = nn.Linear(self.latent_dim, self.latent_dim)
        if cmd_args.rnn_type == 'gru':
            self.gru = nn.GRU(self.latent_dim, 501, 3)
        elif cmd_args.rnn_type == 'sru':
            self.gru = SRU(self.latent_dim, 501, 3)
        else:
            raise NotImplementedError

        self.decoded_logits = nn.Linear(501, DECISION_DIM)
        weights_init(self)

    def forward(self, z):
        assert len(z.size()) == 2 # assert the input is a matrix

        h = self.z_to_latent(z)
        h = F.relu(h)

        rep_h = h.expand(self.max_len, z.size()[0], z.size()[1]) # repeat along time steps

        out, _ = self.gru(rep_h) # run multi-layer gru

        logits = self.decoded_logits(out)

        return logits

class PerpCalculator(nn.Module):
    def __init__(self):
        super(PerpCalculator, self).__init__()

    '''
    input:
        true_binary: one-hot, with size=time_steps x bsize x DECISION_DIM
        rule_masks: binary tensor, with size=time_steps x bsize x DECISION_DIM
        raw_logits: real tensor, with size=time_steps x bsize x DECISION_DIM
    '''
    def forward(self, true_binary, rule_masks, raw_logits):
        if cmd_args.loss_type == 'binary':
            exp_pred = torch.exp(raw_logits) * rule_masks

            norm = F.torch.sum(exp_pred, 2, keepdim=True)
            prob = F.torch.div(exp_pred, norm)

            return F.binary_cross_entropy(prob, true_binary) * cmd_args.max_decode_steps

        if cmd_args.loss_type == 'perplexity':
            return my_perp_loss(true_binary, rule_masks, raw_logits)

        print('unknown loss type %s' % cmd_args.loss_type)
        raise NotImplementedError

if __name__ == '__main__':
    pass

#!/usr/bin/env python


from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = prod(t.size())
        fan_out = prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)

def orthogonal_gru(t):
    assert len(t.size()) == 2
    assert t.size()[0] == 3 * t.size()[1]
    hidden_dim = t.size()[1]

    x0 = torch.Tensor(hidden_dim, hidden_dim)
    x1 = torch.Tensor(hidden_dim, hidden_dim)
    x2 = torch.Tensor(hidden_dim, hidden_dim)

    nn.init.orthogonal(x0)
    nn.init.orthogonal(x1)
    nn.init.orthogonal(x2)

    t[0:hidden_dim, :] = x0
    t[hidden_dim:2*hidden_dim, :] = x1
    t[2*hidden_dim:3*hidden_dim, :] = x2

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.Conv1d):
            p.bias.data.zero_()
            glorot_uniform(p.weight.data)
            print('a Conv1d inited')
        elif isinstance(p, nn.Linear):
            p.bias.data.zero_()
            glorot_uniform(p.weight.data)
            print('a Linear inited')
        elif isinstance(p, nn.GRU):
            for k in range(p.num_layers):
                getattr(p,'bias_ih_l%d'%k).data.zero_()
                getattr(p,'bias_hh_l%d'%k).data.zero_()
                glorot_uniform(getattr(p,'weight_ih_l%d'%k).data)
                orthogonal_gru(getattr(p,'weight_hh_l%d'%k).data)
            print('a GRU inited')
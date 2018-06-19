#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append( '%s/../mol_common' % os.path.dirname(os.path.realpath(__file__)) )
from cmd_args import cmd_args

class MyPerpLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, true_binary, rule_masks, input_logits):
        ctx.save_for_backward(true_binary, rule_masks, input_logits)

        b = F.torch.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) * rule_masks + 1e-30

        norm = torch.sum(exp_pred, 2, keepdim=True)
        prob = torch.div(exp_pred, norm)

        ll = F.torch.abs(F.torch.sum( true_binary * prob, 2))
        
        mask = 1 - rule_masks[:, :, -1]

        logll = mask * F.torch.log(ll)

        loss = -torch.sum(logll) / true_binary.size()[1]
        
        if input_logits.is_cuda:
            return torch.Tensor([loss]).cuda()
        else:
            return torch.Tensor([loss])

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        true_binary, rule_masks, input_logits  = ctx.saved_tensors

        b = F.torch.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) * rule_masks + 1e-30

        norm = torch.sum(exp_pred, 2, keepdim=True)
        prob = torch.div(exp_pred, norm)

        grad_matrix1 = grad_matrix2 = None
        
        grad_matrix3 = prob - true_binary
        bp_mask = rule_masks.clone()
        bp_mask[:, :, -1] = 0

        rescale = 1.0 / true_binary.size()[1]
        grad_matrix3 = grad_matrix3 * bp_mask * grad_output.data * rescale        

        return grad_matrix1, grad_matrix2, Variable(grad_matrix3)

def my_perp_loss(true_binary, rule_masks, raw_logits):    
    return MyPerpLoss.apply(true_binary, rule_masks, raw_logits)

class MyBinaryLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, true_binary, rule_masks, input_logits):
        ctx.save_for_backward(true_binary, rule_masks, input_logits)

        b = F.torch.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) * rule_masks

        norm = torch.sum(exp_pred, 2, keepdim=True)
        prob = torch.div(exp_pred, norm)
                
        loss = F.binary_cross_entropy(prob, true_binary)
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        raise NotImplementedError
        true_binary, rule_masks, input_logits  = ctx.saved_tensors

        b = F.torch.max(input_logits, 2, keepdim=True)[0]
        raw_logits = input_logits - b
        exp_pred = torch.exp(raw_logits) * rule_masks
        # exp_pred = torch.exp(input_logits) * rule_masks
        norm = F.torch.sum(exp_pred, 2, keepdim=True)
        prob = F.torch.div(exp_pred, norm)

        grad_matrix1 = grad_matrix2 = None

        grad_matrix3 = prob - true_binary
        
        grad_matrix3 = grad_matrix3 * rule_masks

        return grad_matrix1, grad_matrix2, Variable(grad_matrix3)

def my_binary_loss(true_binary, rule_masks, raw_logits):
    return MyBinaryLoss.apply(true_binary, rule_masks, raw_logits)

if __name__ == '__main__':
    pass

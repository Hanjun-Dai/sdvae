#!/usr/bin/env python

from __future__ import print_function
from past.builtins import range

import os
import os.path
import sys
import numpy as np
import math
import random

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append('%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

sys.path.append('%s/../prog_vae' % os.path.dirname(os.path.realpath(__file__)))
from prog_vae import ProgVAE, ProgAutoEncoder

sys.path.append('%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)))
import cfg_parser as parser

import h5py

def load_data():
    h5f = h5py.File(cmd_args.data_dump, 'r')
    all_true_binary = h5f['x_train'][:]
    all_rule_masks = h5f['masks_train'][:]
    h5f.close()

    num_val = min([5000, int(all_true_binary.shape[0] * 0.1)])
    return all_true_binary[num_val:], all_rule_masks[num_val:], all_true_binary[0:num_val], all_rule_masks[0:num_val]    

def get_batch_input(selected_idx, data_binary, data_masks, volatile=False):
    true_binary = np.transpose(data_binary[selected_idx, :, :], [1, 0, 2]).astype(np.float32)
    rule_masks = np.transpose(data_masks[selected_idx, :, :], [1, 0, 2]).astype(np.float32)
    x_inputs = np.transpose(true_binary, [1, 2, 0])

    t_vb = torch.from_numpy(true_binary)
    t_ms = torch.from_numpy(rule_masks)

    if cmd_args.mode == 'gpu':
        t_vb = t_vb.cuda()
        t_ms = t_ms.cuda()

    v_tb = Variable(t_vb, volatile=volatile)
    v_ms = Variable(t_ms, volatile=volatile)

    return x_inputs, v_tb, v_ms

def loop_dataset(phase, ae, sample_idxes, data_binary, data_masks, optimizer=None):
    total_loss = []
    pbar = tqdm(range(0, (len(sample_idxes) + (cmd_args.batch_size - 1) * (optimizer is None)) // cmd_args.batch_size), unit='batch')

    if phase == 'train' and optimizer is not None:
        ae.train()
    else:
        ae.eval()

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * cmd_args.batch_size : (pos + 1) * cmd_args.batch_size]
        x_inputs, v_tb, v_ms = get_batch_input(selected_idx, data_binary, data_masks, volatile=(optimizer is None))  # no grad for evaluate mode.

        loss_list = ae(
            x_inputs,
            v_tb,
            v_ms
        )

        perp = loss_list[0].data.cpu().numpy()[0]

        if len(loss_list) == 1: # only perplexity
            loss = loss_list[0]
            kl = 0
        else:
            loss = loss_list[0] + loss_list[1]
            kl = loss_list[1].data.cpu().numpy()[0]

        minibatch_loss = loss.data.cpu().numpy()[0]
        pbar.set_description(' %s loss: %0.5f perp: %0.5f kl: %0.5f' % (phase, minibatch_loss, perp, kl))

        if optimizer is not None:
            assert len(selected_idx) == cmd_args.batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss.append( np.array([minibatch_loss, perp, kl]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss

def main():
    seed = 19260817

    print(cmd_args)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cmd_args.ae_type == 'vae':
        ae = ProgVAE()
    elif cmd_args.ae_type == 'autoenc':
        ae = ProgAutoEncoder()
    else:
        raise Exception('unknown ae type %s' % cmd_args.ae_type)
    if cmd_args.mode == 'gpu':
        ae = ae.cuda()

    if cmd_args.saved_model is not None and cmd_args.saved_model != '':
        if os.path.isfile(cmd_args.saved_model):
            print('loading model from %s' % cmd_args.saved_model)
            ae.load_state_dict(torch.load(cmd_args.saved_model))
    
    assert cmd_args.encoder_type == 'cnn'

    optimizer = optim.Adam(ae.parameters(), lr=cmd_args.learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True, min_lr=0.0001)

    train_binary, train_masks, valid_binary, valid_masks = load_data()
    print('num_train: %d\tnum_valid: %d' % (train_binary.shape[0], valid_binary.shape[0]))

    sample_idxes = list(range(train_binary.shape[0]))
    best_valid_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(sample_idxes)

        avg_loss = loop_dataset('train', ae, sample_idxes, train_binary, train_masks, optimizer)
        print('>>>>average \033[92mtraining\033[0m of epoch %d: loss %.5f perp %.5f kl %.5f' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        if epoch % 1 == 0:
            valid_loss = loop_dataset('valid', ae, list(range(valid_binary.shape[0])), valid_binary, valid_masks)
            print('\033[93m>>>>average valid of epoch %d: loss %.5f perp %.5f kl %.5f\033[0m' % (epoch, valid_loss[0], valid_loss[1], valid_loss[2]))
            valid_loss = valid_loss[0]
            lr_scheduler.step(valid_loss)
            # torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-%d.model' % epoch)
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print('----saving to best model since this is the best valid loss so far.----')
                torch.save(ae.state_dict(), cmd_args.save_dir + '/epoch-best.model')

import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

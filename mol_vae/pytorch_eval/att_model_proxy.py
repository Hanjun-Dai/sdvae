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

from tqdm import tqdm

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

def decode_chunk(raw_logits, use_random, decode_times):
    tree_decoder = create_tree_decoder()    
    chunk_result = [[] for _ in range(raw_logits.shape[1])]
        
    for i in tqdm(range(raw_logits.shape[1])):
        pred_logits = raw_logits[:, i, :]
        walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)

        for _decode in range(decode_times):
            new_t = Node('smiles')
            try:
                tree_decoder.decode(new_t, walker)
                sampled = get_smiles_from_tree(new_t)
            except Exception as ex:
                if not type(ex).__name__ == 'DecodingLimitExceeded':
                    print('Warning, decoder failed with', ex)
                # failed. output a random junk.
                import random, string
                sampled = 'JUNK' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(256))

            chunk_result[i].append(sampled)

    return chunk_result

def batch_decode(raw_logits, use_random, decode_times):
    size = (raw_logits.shape[1] + 7) / 8

    logit_lists = []
    for i in range(0, raw_logits.shape[1], size):
        if i + size < raw_logits.shape[1]:
            logit_lists.append(raw_logits[:, i : i + size, :])
        else:
            logit_lists.append(raw_logits[:, i : , :])

    result_list = Parallel(n_jobs=-1)(delayed(decode_chunk)(logit_lists[i], use_random, decode_times) for i in range(len(logit_lists)))
    return [_1 for _0 in result_list for _1 in _0]

class AttMolProxy(object):
    def __init__(self, *args, **kwargs):
        if cmd_args.ae_type == 'vae':
            self.ae = MolVAE()
        elif cmd_args.ae_type == 'autoenc':
            self.ae = MolAutoEncoder()
        else:
            raise Exception('unknown ae type %s' % cmd_args.ae_type)
        if cmd_args.mode == 'gpu':
            self.ae = self.ae.cuda()

        assert cmd_args.saved_model is not None

	if cmd_args.mode == 'cpu':
		self.ae.load_state_dict(torch.load(cmd_args.saved_model, map_location=lambda storage, loc: storage))
        else:
        	self.ae.load_state_dict(torch.load(cmd_args.saved_model))

        self.onehot_walker = OnehotBuilder()
        self.tree_decoder = create_tree_decoder()
        self.grammar = parser.Grammar(cmd_args.grammar_file)

    def encode(self, chunk, use_random=False):
        '''
        Args:
            chunk: a list of `n` strings, each being a SMILES.

        Returns:
            A numpy array of dtype np.float32, of shape (n, latent_dim)
            Note: Each row should be the *mean* of the latent space distrubtion rather than a sampled point from that distribution.
            (It can be anythin as long as it fits what self.decode expects)
        '''

        '''
        cfg_tree_list = []
        for smiles in chunk:
            ts = parser.parse(smiles, self.grammar)
            assert isinstance(ts, list) and len(ts) == 1

            n = AnnotatedTree2MolTree(ts[0])
            cfg_tree_list.append(n)
        '''
        if type(chunk[0]) is str:
            cfg_tree_list = parse(chunk, self.grammar)
        else:
            cfg_tree_list = chunk
            
        onehot, _ = batch_make_att_masks(cfg_tree_list, self.tree_decoder, self.onehot_walker, dtype=np.float32)

        x_inputs = np.transpose(onehot, [0, 2, 1])
        if use_random:
            self.ae.train()
        else:
            self.ae.eval()
        z_mean, _ = self.ae.encoder(x_inputs)

        return z_mean.data.cpu().numpy()

    def pred_raw_logits(self, chunk, n_steps=None):
        '''
        Args:
            chunk: A numpy array of dtype np.float32, of shape (n, latent_dim)
        Return:
            numpy array of MAXLEN x batch_size x DECISION_DIM
        '''
        if cmd_args.mode == 'cpu':
            z = Variable(torch.from_numpy(chunk))
        else:
            z = Variable(torch.from_numpy(chunk).cuda())

        raw_logits = self.ae.state_decoder(z, n_steps)

        raw_logits = raw_logits.data.cpu().numpy()

        return raw_logits

    def decode(self, chunk, use_random=True):
        '''
        Args:
            chunk: A numpy array of dtype np.float32, of shape (n, latent_dim)
        Return:
            a list of `n` strings, each being a SMILES.
        '''
        raw_logits = self.pred_raw_logits(chunk)

        result_list = []
        for i in range(raw_logits.shape[1]):
            pred_logits = raw_logits[:, i, :]

            walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)

            new_t = Node('smiles')
            try:
                self.tree_decoder.decode(new_t, walker)
                sampled = get_smiles_from_tree(new_t)
            except Exception as ex:
                if not type(ex).__name__ == 'DecodingLimitExceeded':
                    print('Warning, decoder failed with', ex)
                # failed. output a random junk.
                import random, string
                sampled = 'JUNK' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(256))

            result_list.append(sampled)
        
        return result_list

if __name__ == '__main__':
    proxy = AttMolProxy()

    test_list = ['CC1=C(OC2=C1C1=NN(CC(=O)NCC3=CC=CO3)C=C1CC2)C(=O)N1CCOCC1',
                 'C[C@H]1C[C@H](C)CC(C1)NC1=CN=CC(=C1)C1=NN=CN1C']

    z_mean = proxy.encode(test_list)

    print(z_mean.shape)

    decoded_list = proxy.decode(z_mean, use_random=True)
    print('origin: ', test_list)
    print('decode: ', decoded_list)

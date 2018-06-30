#!/usr/bin/env python2

from __future__ import print_function

import sys
import numpy as np
from tqdm import tqdm
import os

sys.path.append('%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

sys.path.append('%s/../prog_vae' % os.path.dirname(os.path.realpath(__file__)))
from prog_vae import ProgVAE, ProgAutoEncoder

sys.path.append('%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)))
import cfg_parser as parser

def main():
    seed = 10960817
    np.random.seed(seed)

    from att_model_proxy import AttProgProxy, batch_decode
    model = AttProgProxy()

    # 0. Constants
    nbprog_per_nbstat = 100
    chunk_size = 100
    encode_times = 10
    decode_times = 25

    # 1. load the progs
    prog_file_patten = '../../dropbox/data/program-free_var_id-check_data/free_var_id-check_data-number-50000-nbstat-%d.test.txt'

    nbstat2prog = {
        nbstat:
        [line.strip() for index, line in zip(xrange(nbprog_per_nbstat), open(prog_file_patten % nbstat).xreadlines())]
        for nbstat in range(cmd_args.min_len, cmd_args.max_len + 1)
    }

    # some routines
    def reconstruct(model, L):
        decode_result = []

        for chunk_start in range(0, len(L), chunk_size):
            chunk = L[chunk_start:chunk_start + chunk_size]
            chunk_result = [[] for _ in range(len(chunk))]
            for _encode in tqdm(range(encode_times)):
                z1 = model.encode(chunk, use_random=True)
                this_encode = []
                for _decode in range(decode_times):
                    _result = model.decode(z1, use_random=True)
                    for index, s in enumerate(_result):
                        chunk_result[index].append(s)

            decode_result.extend(chunk_result)
        assert len(decode_result) == len(L)

        return decode_result

    def save_decode_result(decode_result, L, filename):
        with open(filename, 'w') as fout:
            for s, cand in zip(L, decode_result):
                print('~~~~~'.join([s] + cand), file=fout)

    def cal_accuracy(decode_result, L):
        accuracy = [sum([1 for c in cand if c == s]) * 1.0 / len(cand) for s, cand in zip(L, decode_result)]
        return (sum(accuracy) * 1.0 / len(accuracy))

    # 2. test model

    logfout = open(cmd_args.saved_model + '_reconstruct_result.txt', 'w')
    for nbstat in range(cmd_args.min_len, cmd_args.max_len + 1):
        L = nbstat2prog[nbstat]
        if len(L) == 0:
            continue
        decode_result = reconstruct(model, L)
        save_decode_result(decode_result, L, cmd_args.saved_model + '_reconstruct_nbstat_%d.csv' % nbstat)
        accuracy = cal_accuracy(decode_result, L)
        print('accuracy (nbstat = %d): %f' % (nbstat, accuracy))
        print('accuracy (nbstat = %d): %f' % (nbstat, accuracy), file=logfout)

    logfout.close()


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

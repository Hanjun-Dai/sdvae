#!/usr/bin/env python2

from __future__ import print_function

import os
import sys

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '../prog_data/')
import evaluate

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
    nb_latent_point = 1000
    sample_times = 100
    chunk_size = 100

    def cal_valid_prior(model, latent_dim):
        parser = evaluate.get_parser(cmd_args.grammar_file)

        whole_valid, whole_total = 0, 0

        latent_points = np.random.normal(size=(nb_latent_point, latent_dim)).astype(np.float32)

        raw_logits = model.pred_raw_logits(latent_points)

        result_list = batch_decode(raw_logits, True, sample_times)

        pbar = tqdm(list(range(nb_latent_point)), desc='sampling')

        for _sample in pbar:
            _result = result_list[_sample]
            assert len(_result) == sample_times
            for index, s in enumerate(_result):
                prog = s
                # trying evaluate it
                try:
                    tokens = evaluate.tokenize(prog)
                    tree = evaluate.parse(parser, tokens)
                    if tree is not None:
                        x = 0.12345
                        y, msg = evaluate.eval_at(tree, v0_val=x)
                        if y is not None or (y is None and msg.startswith('runtime error:')):
                            whole_valid += 1
                except ValueError:
                    pass

                whole_total += 1

            pbar.set_description(
                'valid : total = %d : %d = %.5f' % (whole_valid, whole_total, whole_valid * 1.0 / whole_total)
            )

        return 1.0 * whole_valid / whole_total

    # 2. test model

    valid_prior = cal_valid_prior(model, cmd_args.latent_dim)    
    valid_prior_save_file = cmd_args.saved_model + '_valid_prior.txt'
    print('valid prior:', valid_prior)
    with open(valid_prior_save_file, 'w') as fout:
        print('valid prior:', valid_prior, file=fout)


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

#!/usr/bin/env python2

from __future__ import print_function

import sys

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

from rdkit import Chem

# 0. Constants
nb_latent_point = 1000
chunk_size = 100
sample_times = 10

def cal_valid_prior(model, latent_dim):
    from att_model_proxy import batch_decode
    whole_valid, whole_total = 0, 0
    pbar = tqdm(list(range(0, nb_latent_point, chunk_size)), desc='decoding')
    for start in pbar:
        end = min(start + chunk_size, nb_latent_point)
        latent_point = np.random.normal(size=(end - start, latent_dim))
        latent_point = latent_point.astype(np.float32)

        raw_logits = model.pred_raw_logits(latent_point, 1500)
        decoded_array = batch_decode(raw_logits, True, decode_times=sample_times)

        for i in range(end - start):
            for j in range(sample_times):
                s = decoded_array[i][j]
                if not s.startswith('JUNK') and Chem.MolFromSmiles(s) is not None:
                    whole_valid += 1
                whole_total += 1
        pbar.set_description('valid : total = %d : %d = %.5f' % (whole_valid, whole_total, whole_valid * 1.0 / whole_total))
    return 1.0 * whole_valid / whole_total

def main():
    seed = 10960817
    np.random.seed(seed)

    from att_model_proxy import AttMolProxy as ProxyModel
    from att_model_proxy import cmd_args

    model = ProxyModel()

    valid_prior = cal_valid_prior(model, cmd_args.latent_dim)

    valid_prior_save_file =  cmd_args.saved_model + '-valid_prior.txt'
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

#!/usr/bin/env python2

from __future__ import print_function

import sys

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from rdkit import Chem

# 0. Constants
nb_latent_point = 200
chunk_size = 100
sample_times = 100

from collections import Counter

def cal_valid_prior(model, latent_dim):
    from att_model_proxy import batch_decode
    from att_model_proxy import cmd_args
    whole_valid, whole_total = 0, 0
    latent_point = np.random.normal(size=(nb_latent_point, latent_dim))
    latent_point = latent_point.astype(np.float32)

    raw_logits = model.pred_raw_logits(latent_point)
    decoded_array = batch_decode(raw_logits, True, decode_times=sample_times)

    decode_list = []
    for i in range(nb_latent_point):
        c = Counter()
        for j in range(sample_times):
            c[decoded_array[i][j]] += 1
        decoded = c.most_common(1)[0][0]
        if decoded.startswith('JUNK'):
            continue
        m = Chem.MolFromSmiles(decoded)
        if m is None:
            continue
        decode_list.append(decoded)
        if len(decode_list) == 100:
            break

    valid_prior_save_file =  cmd_args.saved_model + '-sampled_prior.txt'
    with open(valid_prior_save_file, 'w') as fout:
        for row in decode_list:
            fout.write('%s\n' % row)
    
def main():
    from att_model_proxy import cmd_args
    seed = cmd_args.seed
    np.random.seed(seed)

    from att_model_proxy import AttMolProxy as ProxyModel
    from att_model_proxy import cmd_args

    model = ProxyModel()

    cal_valid_prior(model, cmd_args.latent_dim)

import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

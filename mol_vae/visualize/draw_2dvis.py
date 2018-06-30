#!/usr/bin/env python

from __future__ import print_function
from past.builtins import range

import os
import sys
import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import math
import random

sys.path.append('%s/../mol_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

sys.path.append('%s/../pytorch_eval' % os.path.dirname(os.path.realpath(__file__)))
from att_model_proxy import AttMolProxy, batch_decode

import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for encoding')
cmd_opt.add_argument('-sample_idx', type=int, help='sample index')
cmd_opt.add_argument('-axisone', type=int, help='frist axis')
cmd_opt.add_argument('-axistwo', type=int, help='second axis')
cmd_opt.add_argument('-grid_size', type=int, default=13, help='print k x k grid')
cmd_opt.add_argument('-gap', type=float, help='gap of interpolation')
cmd_opt.add_argument('-proj_type', type=str, help='axis | proj')
args, _ = cmd_opt.parse_known_args()

from collections import Counter
from tqdm import tqdm

decode_times = 100

if __name__ == '__main__':
    cur_batch_smiles = []
    print(cmd_args)
    print(args)
    seed = 1
    np.random.seed(seed)
    if args.proj_type == 'axis':
        assert args.axisone != args.axistwo

    smiles = None
    with open(cmd_args.smiles_file, 'r') as f:
        idx = 0
        for row in f:
            if idx == args.sample_idx:
                smiles = row.strip()
                break
            idx += 1
    assert smiles is not None

    model = AttMolProxy()

    z = model.encode([smiles], use_random=False)
    mid = args.grid_size // 2
    z_list = []
    
    tmp = np.random.randn(z.shape[1], z.shape[1])
    from scipy.linalg import qr
    Q, R = qr(tmp)
    proj_mat = tmp[0 : 2, :]

    for i in range(args.grid_size):
        for j in range(args.grid_size):            
            if args.proj_type == 'axis':
                z_ij = np.copy(z)
                z_ij[0, args.axisone] += (i - mid) * args.gap
                z_ij[0, args.axistwo] += (j - mid) * args.gap
            elif args.proj_type == 'proj':
                pseudo_inv = np.linalg.pinv(np.dot(proj_mat.T, proj_mat))

                noise = np.array([[(i - mid) * args.gap], [(j - mid) * args.gap]])
                z_noise = np.dot(proj_mat.T, noise)
                z_noise = np.dot(pseudo_inv, z_noise)
                z_ij = z + z_noise.T
            else:
                raise NotImplementedError            
            z_list.append(z_ij.astype(np.float32))
    z_mat = np.vstack(z_list)
    print(z_mat[:, 0 : 2])
    raw_logits = model.pred_raw_logits(z_mat)

    result_list = batch_decode(raw_logits, True, decode_times)

    decode_list = []
    for i in range(z_mat.shape[0]):
        c = Counter()
        for j in range(decode_times):
            c[result_list[i][j]] += 1
        ordered_codes = c.most_common()

        decoded = None
        for p in ordered_codes:
            j = p[0]
            m = Chem.MolFromSmiles(j)
            if m is not None:
                decoded = j
                break
        assert decoded is not None
        decode_list.append(decoded)

    mol_list = []
    for i in range(len(decode_list)):
        m = Chem.MolFromSmiles(decode_list[i])
        assert m is not None
        mol_list.append(m)

    img = Draw.MolsToGridImage(mol_list, molsPerRow=args.grid_size,subImgSize=(200, 200), useSVG=True)

    with open('%s/m-%s-a1-%d-a2-%d-s-%d-g-%.2f.svg' % (cmd_args.save_dir, args.proj_type, args.axisone, args.axistwo, args.sample_idx, args.gap), 'w') as f:
        f.write(img)

    # for i in tqdm(range(10)):
    #     for j in range(len(result_list)):
    #         counters[j][result_list[j]] += 1


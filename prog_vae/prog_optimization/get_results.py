
import pickle
import gzip

from sparse_gp import SparseGP

import scipy.stats as sps

import numpy as np
import sys
import os
sys.path.append('%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

# We define the functions used to load and save objects

def save_object(obj, filename):

    """
    Function that saves an object to a file using pickle
    """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):

    """
    Function that loads an object from a file using pickle
    """

    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()

    return ret


import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for encoding')
cmd_opt.add_argument('-y_norm', type=int, help='normalize target?')
cmd_opt.add_argument('-prog_idx', type=int, help='index of gold program')

args, _ = cmd_opt.parse_known_args()

import glob
if __name__ == '__main__':
    print(cmd_args)
    print(args)
    
    result_list = []
    for seed in range(7, 11):

        for i in range(5):            
            valid_fname = cmd_args.save_dir + '/valid_eq-prog-%d-y-%d-seed-%d-iter-%d.dat' % (args.prog_idx,args.y_norm, seed, i)
            score_fname = cmd_args.save_dir + '/scores-prog-%d-y-%d-seed-%d-iter-%d.dat' % (args.prog_idx,args.y_norm, seed, i)
            progs = np.array(load_object(valid_fname))
            scores = np.array(load_object(score_fname))

            for j in range(len(scores)):
                result_list.append((scores[j], progs[j]))
            
    result_list = sorted(result_list, key=lambda x: x[0])
    prev = -1
    cnt = 0
    for i in range(len(result_list)):
        if result_list[i][0] != prev:
            print(result_list[i][0], result_list[i][1])
            prev = result_list[i][0]
            cnt += 1
            if cnt > 10:
                break

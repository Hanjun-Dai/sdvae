
import pickle
import gzip

from sparse_gp import SparseGP

import scipy.stats as sps

import numpy as np
import sys
import os
sys.path.append('%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

gold_prog_list = []
with open('%s/../prog_data/gold_prog.txt' % os.path.dirname(os.path.realpath(__file__))) as f:
    for row in f:
        gold_prog_list.append(row.strip())


import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for encoding')
cmd_opt.add_argument('-seed', type=int, help='random seed')
cmd_opt.add_argument('-min_len', type=int, help='min # of statements')
cmd_opt.add_argument('-max_len', type=int, help='max # of statements')
cmd_opt.add_argument('-phase', type=str, help='train / test')
cmd_opt.add_argument('-prefix', type=str, help='data prefix')
cmd_opt.add_argument('-data_dir', type=str, help='data folder')
cmd_opt.add_argument('-prog_idx', type=int, help='index of gold program')
cmd_opt.add_argument('-feature_dump', type=str, help='feature numpy dump')
cmd_opt.add_argument('-gp_lr', type=float, help='learning rate of gaussian process')

args, _ = cmd_opt.parse_known_args()

if __name__ == '__main__':
    print(cmd_args)
    print(args)
    np.random.seed(args.seed)

    fmt = args.feature_dump.split('.')[-1]
    if fmt == 'npy':
        X = np.load(args.feature_dump)
    elif fmt == 'txt':
        X = np.loadtxt(args.feature_dump)
    else:
        print('unknown feature dump format ' + fmt)
        raise NotImplementedError
    gold_prog = gold_prog_list[args.prog_idx]

    y = []
    for l in range(args.min_len, args.max_len + 1):
        if args.phase == 'train':
            fname = '%s/%s-number-50000-nbstat-%d.txt.target_for_[%s].txt' % (args.data_dir, args.prefix, l, gold_prog)
        else:
            fname = '%s/%s-number-50000-nbstat-%d.test.txt.target_for_[%s].txt' % (args.data_dir, args.prefix, l, gold_prog)

        cur_scores = np.loadtxt(fname)
        y.append(np.reshape(cur_scores, [-1, 1]))        
    
    y = np.vstack(y)
    
    # y /= np.max(y)
    assert X.shape[0] == y.shape[0]

    n = X.shape[ 0 ]
    permutation = np.random.choice(n, n, replace = False)

    X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
    X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

    y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
    y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

    np.random.seed(0)
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
        y_test, minibatch_size = 10 * M, max_iterations = cmd_args.num_epochs, learning_rate = args.gp_lr)

    with open('%s/sgp-e-%d-seed-%d-lr-%.4f.txt' % (cmd_args.save_dir, cmd_args.num_epochs, args.seed, args.gp_lr), 'w') as f:
        pred, uncert = sgp.predict(X_test, 0 * X_test)
        error = np.sqrt(np.mean((pred - y_test)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
        f.write('Test RMSE: %.10f\n' % error)
        f.write('Test ll: %.10f\n' % testll)
        print 'Test RMSE: ', error
        print 'Test ll: ', testll

        pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        f.write('Train RMSE: %.10f\n' % error)
        f.write('Train ll: %.10f\n' % trainll)        
        print 'Train RMSE: ', error
        print 'Train ll: ', trainll

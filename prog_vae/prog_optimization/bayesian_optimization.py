
import pickle
import gzip

from sparse_gp import SparseGP

import scipy.stats as sps

import numpy as np
import sys
import os
sys.path.append('%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

sys.path.append('%s/../prog_eval' % os.path.dirname(os.path.realpath(__file__)))
from att_model_proxy import AttProgProxy, batch_decode

sys.path.append('%s/../prog_data' % os.path.dirname(os.path.realpath(__file__)))
from bo_target import BOTarget
import evaluate

parser = evaluate.get_parser(cmd_args.grammar_file)

gold_prog_list = []
with open('%s/../prog_data/gold_prog.txt' % os.path.dirname(os.path.realpath(__file__))) as f:
    for row in f:
        gold_prog_list.append(row.strip())

def is_prog_valid(prog):
    tokens = evaluate.tokenize(prog)
    tree = evaluate.parse(parser, tokens)

    if tree is not None:
        x = 0.12345
        y, msg = evaluate.eval_at(tree, v0_val=x)
        if y is not None or (y is None and msg.startswith('runtime error:')):
            return True

    return False

def decode_from_latent_space(latent_points, model):

    decode_attempts = 25
    raw_logits = model.pred_raw_logits(latent_points.astype(np.float32))

    decoded_programs = batch_decode(raw_logits, True, decode_attempts)    
    # We see which ones are decoded by rdkit
    
    rdkit_molecules = []
    for i in range(decode_attempts):
        rdkit_molecules.append([])
        for j in range(latent_points.shape[ 0 ]):
            smile = decoded_programs[ j ][ i ]

            if not is_prog_valid(smile):
                rdkit_molecules[ i ].append(None)
            else:
                rdkit_molecules[ i ].append(smile)

    import collections

    rdkit_molecules = np.array(rdkit_molecules)

    final_smiles = []
    for i in range(latent_points.shape[ 0 ]):

        aux = collections.Counter(rdkit_molecules[ ~np.equal(rdkit_molecules[ :, i ], None) , i ])
        if len(aux) > 0:
            smile = aux.items()[ np.argmax(aux.values()) ][ 0 ]
        else:
            smile = None
        final_smiles.append(smile)

    return final_smiles

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
cmd_opt.add_argument('-seed', type=int, help='random seed')
cmd_opt.add_argument('-min_len', type=int, help='min # of statements')
cmd_opt.add_argument('-max_len', type=int, help='max # of statements')
cmd_opt.add_argument('-y_norm', type=int, help='normalize target?')
cmd_opt.add_argument('-prog_idx', type=int, help='index of gold program')
cmd_opt.add_argument('-phase', type=str, help='train / test')
cmd_opt.add_argument('-prefix', type=str, help='data prefix')
cmd_opt.add_argument('-data_dir', type=str, help='data folder')
cmd_opt.add_argument('-feature_dump', type=str, help='feature numpy dump')
cmd_opt.add_argument('-gp_lr', type=float, help='learning rate of gaussian process')

args, _ = cmd_opt.parse_known_args()
if __name__ == '__main__':
    print(cmd_args)
    print(args)

    model = AttProgProxy()

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
   
    if args.y_norm: 
        y_mean = np.mean(y)
        y_std = np.std(y)
        y = (y - y_mean) / y_std
    # y /= np.max(y)
    assert X.shape[0] == y.shape[0]

    n = X.shape[ 0 ]
    permutation = np.random.choice(n, n, replace = False)

    X_train = X[ permutation, : ][ 0 : np.int(np.round(0.99 * n)), : ]
    X_test = X[ permutation, : ][ np.int(np.round(0.99 * n)) :, : ]

    y_train = y[ permutation ][ 0 : np.int(np.round(0.99 * n)) ]
    y_test = y[ permutation ][ np.int(np.round(0.99 * n)) : ]

    bo_target = BOTarget(parser, gold_prog=gold_prog)

    for iteration in range(5):
        print(iteration)
        np.random.seed(args.seed * iteration)
        M = 500

        sgp = SparseGP(X_train, 0 * X_train, y_train, M)
        sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
          y_test, minibatch_size = 10 * M, max_iterations = cmd_args.num_epochs, learning_rate = args.gp_lr)


        next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))
        valid_eq_final = decode_from_latent_space(next_inputs, model)
        
        new_features = next_inputs
        
        save_object(valid_eq_final, "%s/valid_eq-prog-%d-y-%d-seed-%d-iter-%d.dat" % (cmd_args.save_dir, args.prog_idx, args.y_norm, args.seed, iteration))

        scores = []

        for i in range(len(valid_eq_final)):
            if valid_eq_final[ i ] is not None:
                score = bo_target(valid_eq_final[i])
            else:
                score = np.log(1+BOTarget.WORST)

            scores.append(score)
            print(i)

        print(valid_eq_final)
        print(scores)
        save_object(scores, "%s/scores-prog-%d-y-%d-seed-%d-iter-%d.dat" % (cmd_args.save_dir, args.prog_idx, args.y_norm, args.seed, iteration))

        if args.y_norm:
            scores = (np.array(scores) - y_mean) / y_std
        # scores = np.array(scores) / np.max(y)
        if len(new_features) > 0:
            X_train = np.concatenate([ X_train, new_features ], 0)
            y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

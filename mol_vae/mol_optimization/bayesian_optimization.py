
import pickle
import gzip

from sparse_gp import SparseGP

import scipy.stats as sps

import numpy as np
import sys
import os
sys.path.append('%s/../mol_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

sys.path.append('%s/../pytorch_eval' % os.path.dirname(os.path.realpath(__file__)))
from att_model_proxy import AttMolProxy, batch_decode

import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for encoding')
cmd_opt.add_argument('-seed', type=int, help='random seed')
cmd_opt.add_argument('-feature_dump', type=str, help='feature numpy dump')
cmd_opt.add_argument('-zinc_dir', type=str, help='dir contains target, logp, cycle, sa')
cmd_opt.add_argument('-gp_lr', type=float, help='learning rate of gaussian process')

args, _ = cmd_opt.parse_known_args()

def decode_from_latent_space(latent_points, model):
    decode_attempts = 500
    raw_logits = model.pred_raw_logits(latent_points.astype(np.float32))
    decoded_molecules = batch_decode(raw_logits, True, decode_attempts)

    rdkit_molecules = []
    for i in range(decode_attempts):
        rdkit_molecules.append([])
        for j in range(latent_points.shape[ 0 ]):
            smile = decoded_molecules[ j ][ i ]
            if MolFromSmiles(smile) is None:
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


from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw
# import image
import copy
import time

from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
import sascorer
import networkx as nx
from rdkit.Chem import rdmolops

if __name__ == '__main__':
    print(cmd_args)
    print(args)
    model = AttMolProxy()

    logP_values = np.loadtxt(args.zinc_dir + '/logP_values.txt')
    SA_scores = np.loadtxt(args.zinc_dir + '/SA_scores.txt')
    cycle_scores = np.loadtxt(args.zinc_dir + '/cycle_scores.txt')
    SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
    cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

    targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized

    np.random.seed(args.seed)

    fmt = args.feature_dump.split('.')[-1]
    if fmt == 'npy':
        X = np.load(args.feature_dump)
    elif fmt == 'txt':
        X = np.loadtxt(args.feature_dump)
    else:
        print('unknown feature dump format ' + fmt)
        raise NotImplementedError
    
    y = -np.loadtxt(args.zinc_dir + '/targets.txt')
    y = y.reshape((-1, 1))
    n = X.shape[ 0 ]
    permutation = np.random.choice(n, n, replace = False)

    X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
    X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

    y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
    y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

    iteration = 0
    while iteration < 5:

        np.random.seed(iteration * args.seed)
        M = 500
        sgp = SparseGP(X_train, 0 * X_train, y_train, M)
        sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
            y_test, minibatch_size = 10 * M, max_iterations = cmd_args.num_epochs, learning_rate = args.gp_lr)
    
        # pred, uncert = sgp.predict(X_test, 0 * X_test)
        # error = np.sqrt(np.mean((pred - y_test)**2))
        # testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
        # print 'Test RMSE: ', error
        # print 'Test ll: ', testll

        # pred, uncert = sgp.predict(X_train, 0 * X_train)
        # error = np.sqrt(np.mean((pred - y_train)**2))
        # trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        # print 'Train RMSE: ', error
        # print 'Train ll: ', trainll    

        next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))        
        valid_smiles_final = decode_from_latent_space(next_inputs, model)
        save_object(valid_smiles_final, "%s/valid_smiles-seed-%d-iter-%d.dat" % (cmd_args.save_dir, args.seed, iteration))
        new_features = next_inputs

        scores = []
        for i in range(len(valid_smiles_final)):
            if valid_smiles_final[ i ] is not None:
                current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles_final[ i ]))
                current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles_final[ i ]))
                cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles_final[ i ]))))
                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([ len(j) for j in cycle_list ])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6

                current_cycle_score = -cycle_length

                current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
                current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
                current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

                score = (current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized)
            else:
                score = -max(y)[ 0 ]

            scores.append(-score)
            print(i)
        
        print(valid_smiles_final)
        print(scores)
        save_object(scores, "%s/scores-seed-%d-iter-%d.dat" % (cmd_args.save_dir, args.seed, iteration))
        if len(new_features) > 0:
            X_train = np.concatenate([ X_train, new_features ], 0)
            y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
        
        iteration += 1
        print(iteration)
    # with open('%s/sgp-e-%d-seed-%d-lr-%.4f.txt' % (cmd_args.save_dir, cmd_args.num_epochs, args.seed, args.gp_lr), 'w') as f:

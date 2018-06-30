import sys
import os
import rdkit
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Fingerprints import FingerprintMols

def get_fp_list(fname, fp_func_list):
    fp_list = []
    with open(fname, 'r') as f:
        for row in f:
            smiles = row.strip()
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                print smiles
            fps = []
            for func in fp_func_list:
                fps.append(func(m))

            fp_list.append(fps)
    return fp_list

import numpy as np
def eval_similarity(fp_list, dim, evaluator):
    s_list = []
    for i in range(len(fp_list) - 1):
        for j in range(i + 1, len(fp_list)):
            s_list.append(evaluator(fp_list[i][dim], fp_list[j][dim]))
    s_list = np.array(s_list) 
    return np.mean(s_list), np.std(s_list)

if __name__ == '__main__':
    f = sys.argv[1]
    fp_func_list = [lambda x : AllChem.GetMorganFingerprint(x,2), 
                    lambda x: MACCSkeys.GenMACCSKeys(x),
                    lambda x: Pairs.GetAtomPairFingerprint(x), 
                    lambda x: FingerprintMols.FingerprintMol(x)]

    evaluators = [lambda x, y: DataStructs.DiceSimilarity(x, y), 
                  lambda x, y: DataStructs.FingerprintSimilarity(x, y), 
                  lambda x, y: DataStructs.DiceSimilarity(x, y), 
                  lambda x, y: DataStructs.FingerprintSimilarity(x, y)]

    fp_list = get_fp_list(f, fp_func_list)

    for i in range(len(fp_func_list)):
        m, s = eval_similarity(fp_list, i, evaluators[i])
        print(1 - m, s)

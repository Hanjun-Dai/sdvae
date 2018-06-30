#!/usr/bin/env python2

from __future__ import print_function

from collections import Counter
import os
import sys

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm


sys.path.append('%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args

sys.path.append('%s/../prog_data' % os.path.dirname(os.path.realpath(__file__)))
import evaluate

prog_a = "v6=cos(7);v8=exp(9);v2=v8*v0;v9=v2/v6;return:v9"
prog_b = "v6=exp(v0);v8=v6-4;v4=4*v8;v7=v4+v8;return:v7"
interplotation_number = 10
decode_try_number = 50

if __name__ == '__main__':
    seed = 10960817
    np.random.seed(seed)

    from att_model_proxy import AttProgProxy, batch_decode
    model = AttProgProxy()

    z_a = model.encode([prog_a], use_random=False)
    z_b = model.encode([prog_b], use_random=False)

    z_d = z_b - z_a

    interplotation = []
    interplotation.append(prog_a)

    for i in range(1, interplotation_number):
        z = z_a + z_d / float(interplotation_number) * float(i)
        z = np.tile(z, (decode_try_number, 1))
        s = model.decode(z, use_random=True)
        s = Counter(s).most_common(1)[0][0]
        interplotation.append(s)
    interplotation.append(prog_b)

    for prog in interplotation:
        print(prog)

    interplotation_save_file = cmd_args.save_dir + '/interplotation.txt'
    with open(interplotation_save_file, 'w') as fout:
        for prog in interplotation:
            print(prog, file=fout)
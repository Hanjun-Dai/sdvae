#!/usr/bin/env python2

from __future__ import print_function

import math
import os
import random
import sys

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

import evaluate
import bo_target as bo_target_module

import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_string('prog_file', '', 'the prog file')
gflags.DEFINE_string('gold_prog_file', '', 'the file for list of gold progs as reference.')
gflags.DEFINE_integer('n_jobs', -1, 'nubmer of jobs. -1 for all.')


def do_many(bo_target, prog_list):
    result = [bo_target(prog) for prog in prog_list]
    return result


def main():
    FLAGS(sys.argv)

    random.seed(19260817)
    cfg_grammar_file = (
        os.path.dirname(os.path.realpath(__file__)) + '/../../dropbox/context_free_grammars/prog_leftskew.grammar'
    )

    prog_file = FLAGS.prog_file
    n_jobs = FLAGS.n_jobs

    # 1. reading
    print('reading progs...')
    prog_list = [_.strip() for _ in open(prog_file).readlines()]

    # 2. compute
    gold_prog_list = [_.strip() for _ in open(FLAGS.gold_prog_file).readlines() if _.strip() != '']

    for gold_prog in gold_prog_list:
        print('producing bo_target for [%s]...' % gold_prog)

        target_file = FLAGS.prog_file + '.target_for_[%s].txt' % gold_prog
        # simple_target_file = FLAGS.prog_file + '.simple_target_for_[%s].txt' % gold_prog

        parser = evaluate.get_parser(cfg_grammar_file)
        bo_target = bo_target_module.BOTarget(parser, gold_prog)

        block_size = 1000
        block_result = Parallel(
            n_jobs=n_jobs, verbose=50
        )(
            delayed(do_many)(bo_target, prog_list[start:start + block_size])
            for start in range(0, len(prog_list), block_size)
        )

        result = [_2 for _1 in block_result for _2 in _1]

        # 3. saving
        print('saving...')
        with open(target_file, 'w') as fout:
            for v in result:
                print(v, file=fout)

        #with open(simple_target_file, 'w') as fout:
        #    for prog in prog_list:
        #        print(len(prog.split(';')), file=fout)



import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

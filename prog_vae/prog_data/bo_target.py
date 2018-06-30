#!/usr/bin/env python2

from __future__ import print_function

import math
import os
import random

import numpy as np

import evaluate


class BOTarget(object):

    WORST = math.exp(16)    # magic

    def __init__(self, parser, gold_prog):
        self.parser = parser
        self.gold_prog = gold_prog
        v0_range = 5.
        self.v0_val_list = np.linspace(-v0_range, v0_range, num=1000)

        gold_tree = self.prog_to_tree(gold_prog)
        assert gold_tree is not None
        self.gold_prog_y_many = evaluate.eval_at_many(gold_tree, self.v0_val_list)

        assert self.calc_target(self.gold_prog_y_many) is not None

    def calc_target(self, a, b=None):
        assert isinstance(a, list)
        if b is None:
            b = [0.] * len(a)
        assert len(b) == len(a)

        WORST = self.WORST
        try:
            c = [_a - _b for _a, _b in zip(a, b)]
            c = np.array(c)
            score = np.log(1 + np.mean(np.minimum(c ** 2, WORST)))
        except:
            score = np.log(1 + WORST)
        if not np.isfinite(score):
            score = np.log(1 + WORST)

        return score

    def prog_to_tree(self, prog):
        tokens = evaluate.tokenize(prog)
        tree = evaluate.parse(self.parser, tokens)
        return tree

    def __call__(self, prog):
        prog_tree = self.prog_to_tree(prog)
        prog_y_many = evaluate.eval_at_many(prog_tree, self.v0_val_list)
        target = self.calc_target(self.gold_prog_y_many, prog_y_many)
        return target


def main():
    random.seed(19260817)
    cfg_grammar_file = (
        os.path.dirname(os.path.realpath(__file__)) + '/../../dropbox/context_free_grammars/prog_leftskew.grammar'
    )

    parser = evaluate.get_parser(cfg_grammar_file)

    # bo_target = BOTarget(parser, gold_prog='v1=1-v0;v2=v0*v1;v3=exp(v0);v4=v2+v3;return:v4')
    bo_target = BOTarget(parser, gold_prog='v1=exp(v0);v2=exp(v1);return:v2')

    for i in range(20):
        prog = evaluate.gen_one(5)
        print('prog:', prog)
        print('target:', bo_target(prog))


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

#!/usr/bin/env python2

from __future__ import print_function

import math
import os
import random
import sys

from tqdm import tqdm

import evaluate as evaluate_module

import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('nb_stat', 10, 'maximal number of statentment in a program.')
gflags.DEFINE_integer('number', 100000, 'number of programs.')
gflags.DEFINE_integer('seed', 19260817, 'random seed.')
gflags.DEFINE_integer('repeat', 5, 'repeat this times if avaible progs with nb_stat is not enough.')
gflags.DEFINE_string('save_file', '', 'the (train) file to save')
gflags.DEFINE_string('save_test_file', '', 'the test file to save')
gflags.DEFINE_boolean('check_data', False, 'Whether to check (evaluate) the generated data and discard bad ones.')
gflags.DEFINE_boolean('free_var_id', False, 'Whether to use free var_id.')

# global objects
cfg_grammar_file = (
    os.path.dirname(os.path.realpath(__file__)) + '/../../dropbox/context_free_grammars/prog_leftskew.grammar'
)
parser = evaluate_module.get_parser(cfg_grammar_file)
tokenize = evaluate_module.tokenize
parse = evaluate_module.parse
eval_at_many = evaluate_module.eval_at_many


def gen_one(nb_stat):
    def _get_immediate_number():
        return random.choice(list('123456789'))

    def _get_operand(head_set):
        if random.choice(['var', 'immediate_number']) == 'var':
            return random.choice(list(head_set))
        else:
            return _get_immediate_number()

    def _is_var(s):
        return s.startswith('v')

    def _get_current_head_set(*operand_list):
        return set([_ for _ in operand_list if _is_var(_)])

    possible_head_set = set(['v%d' % d for d in range(10)])

    stat_list = []
    used_head_set = set(['v0'])
    free_head_set = set(['v0'])

    for stat_id in range(nb_stat):
        is_final = (stat_id == nb_stat - 1)

        if is_final:
            assert len(free_head_set) == 1  # must have only one...
            stat = 'return:' + list(free_head_set)[0]
        else:
            if FLAGS.free_var_id:
                lhs = random.choice(list(possible_head_set - used_head_set))
            else:
                lhs = 'v%d' % (stat_id + 1)

            while True:
                expr = random.choice(['unary_expr', 'binary_expr'])
                if expr == 'unary_expr':
                    unary_head = random.choice(['-', 'sin', 'cos', 'exp'])
                    operand = _get_operand(used_head_set)
                    if unary_head in ['-']:
                        back_rhs = unary_head + operand
                    else:
                        back_rhs = unary_head + '(' + operand + ')'

                    next_used_head_set = used_head_set | set([lhs])
                    next_free_head_set = (free_head_set - _get_current_head_set(operand)) | set([lhs])
                    if len(next_free_head_set) <= nb_stat - 1 - stat_id:
                        used_head_set = next_used_head_set
                        free_head_set = next_free_head_set
                        break
                elif expr == 'binary_expr':
                    operand_1 = _get_operand(used_head_set)
                    operand_2 = _get_operand(used_head_set)
                    binary_op = random.choice(['+', '-', '*', '/'])
                    if len(_get_current_head_set(operand_1, operand_2)) == 0:
                        continue    # fail. neet at least one var.
                    if binary_op in ['-'] and operand_1 == operand_2:
                        continue    #  lead to zero, bad.
                    if binary_op in ['/'] and operand_2 == '0':
                        continue    #  div by zero, bad.
                    back_rhs = operand_1 + binary_op + operand_2

                    next_used_head_set = used_head_set | set([lhs])
                    next_free_head_set = (free_head_set - _get_current_head_set(operand_1, operand_2)) | set([lhs])
                    if len(next_free_head_set) <= nb_stat - 1 - stat_id:
                        used_head_set = next_used_head_set
                        free_head_set = next_free_head_set
                        break

            stat = lhs + '=' + back_rhs

        stat_list.append(stat)

    prog = ';'.join(stat_list)

    if FLAGS.check_data:
        v0_val_list = [0.] + [1.,2.,3.,4.,5.] + [-1.,-2.,-3.,-4.,-5.] + [1e-5, -1e-5, 1-1e-5,0.12345]
        tokens = tokenize(prog)
        tree = parse(parser, tokens)
        res = eval_at_many(tree, v0_val_list, one_fail_is_enough=True)
        if any([_ is None for _ in res]):
            '''
            print('prog %s failed for v0 = %s' % (prog,
                [v0_val for v0_val, this_res in zip(v0_val_list, res) if this_res is None]
            ))
            '''
            return None    # failed data.
    return prog


def main():
    FLAGS(sys.argv)
    random.seed(FLAGS.seed)

    seen_prog_set = set()

    count_try = 0
    prog_list = []
    fail_ratio = 14    # magic here, never bother....

    if FLAGS.nb_stat == 1:
        number = 1
    elif FLAGS.nb_stat == 2:
        number = min(FLAGS.number, 7200)   # an (human) estimation of upper bound
    else:
        number = FLAGS.number
    for i in tqdm(range(1 if FLAGS.nb_stat == 1 else number)):
        count_try_threshold = max(len(prog_list) * fail_ratio, 100)
        while True:
            prog = gen_one(FLAGS.nb_stat)
            count_try += 1
            if prog is not None and prog not in seen_prog_set:
                seen_prog_set.add(prog)
                break
            if count_try >= count_try_threshold:
                break    # no hope. leave.

        if count_try >= count_try_threshold:
            break    # no hope. leave.
        prog_list.append(prog)

    size_test = min(500, int(len(prog_list) * 0.1))
    prog_test_list = prog_list[:size_test]
    prog_train_list = prog_list[size_test:]

    def ceil_int_div(a, b):
        return (a + b - 1) // b

    expected_size_train = number - size_test
    if len(prog_train_list) < expected_size_train:
        actual_repeat = min(FLAGS.repeat, ceil_int_div(expected_size_train, len(prog_train_list)))
        prog_train_list = (prog_train_list * actual_repeat)[:expected_size_train]

    with open(FLAGS.save_file, 'w') as fout:
        for prog in prog_train_list:
            print(prog, file=fout)

    with open(FLAGS.save_test_file, 'w') as fout:
        for prog in prog_test_list:
            print(prog, file=fout)


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

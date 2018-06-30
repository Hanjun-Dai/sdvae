#!/usr/bin/env python2

from __future__ import print_function

import copy
import math
import os
import random
import re
import sys

import nltk
import numpy as np
import six
from tqdm import tqdm



def get_parser(production_file):
    prods = [_.strip() for _ in open(production_file).readlines()] + ['Nothing -> None']
    string = '\n'.join(prods)
    GCFG = nltk.CFG.fromstring(string)
    parser = nltk.ChartParser(GCFG)
    return parser


def parse(parser, tokens):
    try:
        for x in parser.parse(tokens):
            return x
        return None
    except:
        return None


def tokenize(prog):
    s = prog
    funcs = ['sin', 'cos', 'exp']
    for fn in funcs:
        s = s.replace(fn + '(', fn + ' ')
    s = re.sub(r'([^a-z ])', r' \1 ', s)
    for fn in funcs:
        s = s.replace(fn, fn + ' (')
    return s.split()


class EvaluationError(Exception):
    pass


def eval_at(t, v0_val):

    RETURN_KEY = 'return_val'

    def _get_var_id(t, ctx):
        assert t.label() == 'var_id'
        return t[0]

    def _get_var(t, ctx):
        assert t.label() == 'var'
        return t[0] + _get_var_id(t[1], ctx)

    def _get_lhs(t, ctx):
        assert t.label() == 'lhs'
        var = _get_var(t[0], ctx)
        return var

    def _get_immediate_number(t, ctx):
        return float(t[0][0])

    def _get_operand(t, ctx):
        assert t.label() == 'operand'
        if t[0].label() == 'var':
            var = _get_var(t[0], ctx)
            if var not in ctx:
                raise EvaluationError('refer unassigned var %s' % var)
            return ctx[var]
        elif t[0].label() == 'immediate_number':
            return _get_immediate_number(t[0], ctx)
        else:
            assert False

    unary_op2func = {
        '-': (lambda a: -a),
    }

    unary_func2func = {'sin': math.sin, 'cos': math.cos, 'exp': math.exp}

    def _get_unary_expr(t, ctx):
        assert t.label() == 'unary_expr'
        if t[0].label() == 'unary_op':
            op = t[0][0]
            func = unary_op2func[op]
            operand = _get_operand(t[1], ctx)
            return func(operand)
        elif t[0].label() == 'unary_func':
            func = t[0][0]
            func = unary_func2func[func]
            operand = _get_operand(t[2], ctx)
            return func(operand)
        assert False

    binary_op2func = {
        '+': (lambda a, b: a + b),
        '-': (lambda a, b: a - b),
        '*': (lambda a, b: a * b),
        '/': (lambda a, b: a / b),
    }

    def _get_binary_expr(t, ctx):
        assert t.label() == 'binary_expr'
        op = t[1][0]
        assert op in binary_op2func
        func = binary_op2func[op]
        operand_1 = _get_operand(t[0], ctx)
        operand_2 = _get_operand(t[2], ctx)
        return func(operand_1, operand_2)

    def _get_expr(t, ctx):
        assert t.label() == 'expr'
        if t[0].label() == 'unary_expr':
            return _get_unary_expr(t[0], ctx)
        if t[0].label() == 'binary_expr':
            return _get_binary_expr(t[0], ctx)
        assert False

    def _get_rhs(t, ctx):
        assert t.label() == 'rhs'
        return _get_expr(t[0], ctx)

    def _eval_step(t, ctx):
        label = t.label()
        if label == 'program':
            _eval_step(t[0], ctx)
        elif label == 'stat_list':
            _eval_step(t[0], ctx)
            if len(t) == 3:
                _eval_step(t[2], ctx)
        elif label == 'stat':
            _eval_step(t[0], ctx)
        elif label == 'assign_stat':
            lhs = _get_lhs(t[0], ctx)
            rhs = _get_rhs(t[2], ctx)
            if lhs in ctx:
                raise EvaluationError('multiple assign to a single var %s' % lhs)
            ctx[lhs] = rhs
        elif label == 'return_stat':
            lhs = _get_lhs(t[2], ctx)
            if lhs not in ctx:
                raise EvaluationError('refer unassigned var %s' % lhs)
            if RETURN_KEY in ctx:
                raise EvaluationError('multiple return')
            ctx[RETURN_KEY] = ctx[lhs]

    try:
        ctx = {'v0': v0_val}
        _eval_step(t, ctx)
    except EvaluationError as e:
        return None, 'semantic error: ' + str(e)
    except OverflowError as e:
        return None, 'runtime error: math overflow'
    except ZeroDivisionError as e:
        return None, 'runtime error: division by zero'
    except ValueError as e:
        return None, 'runtime error: ' + str(e)
    except Exception as e:
        return None, 'runtime error: ' + str(e)

    return_val = ctx.get(RETURN_KEY, None)
    if return_val is not None:
        return return_val, ''
    else:
        return None, 'semantic error: ' + 'no return'


def eval_at_many(t, v0_val_list, one_fail_is_enough=False):
    '''
    Returns None if something is really wrong.
    '''

    all_res = []
    for v0_val in v0_val_list:
        res, msg = eval_at(t, v0_val)
        if res is None and one_fail_is_enough:
            return all_res + [None]  # just return what we have so far
        if res is None and msg.startswith('semantic error:'):
            return [None] * len(v0_val_list)
        all_res.append(res)
    return all_res


def demo_eval_at(parser):
    from generate_data import gen_one
    for i in range(1000):
        prog = gen_one(10)
        tokens = tokenize(prog)
        tree = parse(parser, tokens)

        print('evaluate:', prog)
        for x in [0.1234, 1.234, 12.34, 123.4, 1234., 12345.]:
            y, msg = eval_at(tree, v0_val=x)
            print(x, y, msg)
        print('-' * 80)


def demo_eval_at_many(parser):
    from generate_data import gen_one
    v0_val_list = np.linspace(-10.0, 10.0, num=1000)
    for i in range(20):
        prog = gen_one(5)
        tokens = tokenize(prog)
        tree = parse(parser, tokens)
        res = eval_at_many(tree, v0_val_list)

        print('evaluate:', prog)
        if all([_ is None for _ in res]):
            print('all is None!')
            raise Exception
        else:
            target = [math.log(1 + (_ * _)) for _ in res if _ is not None]
            print('target:', np.mean(target))


def main():
    random.seed(19260817)
    cfg_grammar_file = (
        os.path.dirname(os.path.realpath(__file__)) + '/../../dropbox/context_free_grammars/prog_leftskew.grammar'
    )
    parser = get_parser(cfg_grammar_file)

    # demo_eval_at(parser)
    demo_eval_at_many(parser)


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

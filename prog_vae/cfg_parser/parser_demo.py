#!/usr/bin/env python2

import os

import nltk

import cfg_parser as parser


def main():
    cfg_grammar_file = '../../dropbox/context_free_grammars/prog_leftskew.grammar'

    grammar = parser.Grammar(cfg_grammar_file)
    ts = parser.parse(
        'v1=sin(v0);v2=v0*4;v3=v1/v2;v4=cos(v0);v5=v0*3;v6=sin(v1);v7=v3-v6;v8=v7+v5;v9=v8+v4;return:v9', grammar
    )
    t = ts[0]

    print('(ugly) tree:')
    print(t)
    print()

    print('for root:')
    print(
        'symbol is %s, is it non-terminal = %s, it\' value is %s (of type %s)' %
        (t.symbol, isinstance(t, parser.Nonterminal), t.symbol.symbol(), type(t.symbol.symbol()))
    )
    print(
        'rule is %s, its left side is %s (of type %s), its right side is %s, a tuple '
        'which each element can be either str (for terminal) or Nonterminal (for nonterminal)' % (
            t.rule,
            t.rule.lhs(),
            type(t.rule.lhs()),
            t.rule.rhs(),
        )
    )


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

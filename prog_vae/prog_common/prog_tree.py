#!/usr/bin/env python

from prog_util import prod, rule_ranges, TOTAL_NUM_RULES, MAX_VARS, DECISION_DIM
import numpy as np

import sys
import os
sys.path.append( '%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)) )
import cfg_parser
from cmd_args import cmd_args

class Node(object):
    def __init__(self,s, father = None):
        self.symbol = s
        self.children = []
        self.rule_used = None
        self.father = father
        if self.father is None:
            assert self.symbol == 'program'

    def is_created(self):
        if self.rule_used is None:
            return False
        return len(prod[self.symbol][self.rule_used]) == len(self.children)

    def add_child(self, child, pos = None):
        if self.is_created():
            return
        if pos is None:
            self.children.append(child)
        else:
            self.children.insert(pos, child)

def dfs(node, result):
    if len(node.children):
        for c in node.children:
            dfs(c, result)
    else:
        assert node.symbol[0] == node.symbol[-1] == '\''
        result.append(node.symbol[1:-1])

def get_program_from_tree(root):
    result = []
    dfs(root, result)
    st = ''.join(result)
    return st

def _AnnotatedTree2ProgTree(annotated_root, father):
    n = Node(str(annotated_root.symbol), father=father)
    n.rule_used = annotated_root.rule_selection_id
    for c in annotated_root.children:
        new_c = _AnnotatedTree2ProgTree(c, n)
        n.children.append(new_c)

    if n.symbol == 'var':
        assert len(n.children)
        d = n.children[-1]
        assert d.symbol == 'var_id'
        st = d.children[0].symbol
        assert len(st) == 3
        idx = int(st[1 : -1])
        n.var_id = idx
    if isinstance(annotated_root.symbol, cfg_parser.Nonterminal): # it is a non-terminal
        assert len(n.children)
        assert n.is_created()
    else:
        assert isinstance(annotated_root.symbol, str)
        assert len(n.symbol) < 3 or (n.symbol[0] != '\'' and n.symbol[-1] != '\'')        
        n.symbol = '\'' + n.symbol + '\''
    return n

def AnnotatedTree2ProgTree(annotated_root):
    ans = _AnnotatedTree2ProgTree(annotated_root, father=None)
    return ans

if __name__ == '__main__':

    smiles = 'OSC'
    grammar = cfg_parser.Grammar(cmd_args.grammar_file)


    ts = cfg_parser.parse(smiles, grammar)
    assert isinstance(ts, list) and len(ts) == 1

    print(AnnotatedTree2RuleIndices(ts[0]))

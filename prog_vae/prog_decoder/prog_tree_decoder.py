#!/usr/bin/env python


from __future__ import print_function

import os
import sys
import csv
import numpy as np
import math
import random
from collections import defaultdict

sys.path.append( '%s/../prog_common' % os.path.dirname(os.path.realpath(__file__)) )
from cmd_args import cmd_args
from prog_util import prod, MAX_VARS, rule_ranges, MAX_NUM_STATEMENTS, DECISION_DIM
from prog_tree import Node

from tree_walker import ProgramOnehotBuilder

class ProgTreeDecoder(object):
    def __init__(self):
        self.full_var_set = set(range(MAX_VARS))
        self.reset_state()

    def reset_state(self):
        self.num_statements = 0
        self.return_used = False
        self.defined_vars = set([0])

    def get_node(self, node, new_sym, pos):
        if node.is_created():
            assert pos < len(node.children)
            ans = node.children[pos]
            assert ans.symbol == new_sym
            return ans
        return Node(new_sym, node)

    def rand_rule(self, node, sub_ranges = None):
        g_range = rule_ranges[node.symbol]
        idxes = np.arange(g_range[0], g_range[1])
        if sub_ranges is not None:
            idxes = idxes[sub_ranges]

        assert len(idxes)
        if len(idxes) == 1:
            result = 0
        else:
            result = self.walker.sample_index_with_mask(node, idxes)

        if sub_ranges is not None:
            new_idx = sub_ranges[result]
        else:
            new_idx = result
        
        if node.rule_used is not None:
            assert node.rule_used == new_idx
        else:
            node.rule_used = new_idx
            
        return node.rule_used

    def rand_att(self, node, candidates):
        if len(candidates) == 1:
            att_idx = candidates[0]
        else:
            att_idx = self.walker.sample_att(node, candidates)
        if not hasattr(node, 'var_id'):
            node.var_id = att_idx
        else:
            assert node.var_id == att_idx
        return att_idx

    def get_inh_attr(self, attr_dict, key):
        assert attr_dict is not None
        assert key in attr_dict
        return attr_dict[key]

    def tree_generator(self, node, inherit_atts = None):
        if node.symbol == 'stat_list':        
            candidates = [0]
            self.num_statements += 1
            if self.num_statements < MAX_NUM_STATEMENTS:
                candidates.append(1)

            rule = self.rand_rule(node, candidates)
                            
            if rule == 1: # state_list -> stat_list stat_sep stat
                s_list = self.get_node(node, 'stat_list', 0)
                node.add_child(s_list)
                self.tree_generator(s_list, inherit_atts={'is_last' : False})

                sep = self.get_node(node, '\';\'', 1)
                node.add_child(sep)

            s = self.get_node(node, 'stat', -1)
            node.add_child(s)
            if inherit_atts is not None:
                is_last = self.get_inh_attr(inherit_atts, 'is_last')
            else:
                is_last = True
            self.tree_generator(s, inherit_atts={'is_last' : is_last})                        
        elif node.symbol == 'stat':            
            rule = int(self.get_inh_attr(inherit_atts, 'is_last'))
            p = prod[node.symbol][rule]
            s = self.get_node(node, p[0], 0)
            node.add_child(s)
            self.tree_generator(s)
        elif node.symbol == 'assign_stat':
            rhs = self.get_node(node, 'rhs', 2)
            self.tree_generator(rhs, {'is_reuse' : True})

            lhs = self.get_node(node, 'lhs', 0)
            self.tree_generator(lhs, {'is_reuse': False})

            node.add_child(lhs)
            e = self.get_node(node, '\'=\'', 1)
            node.add_child(e)
            node.add_child(rhs)
        elif node.symbol == 'return_stat':
            self.return_used = True
            lhs = self.get_node(node, 'lhs', 2)
            self.tree_generator(lhs, {'is_reuse' : True})

            r = self.get_node(node, '\'return\'', 0)
            node.add_child(r)
            q = self.get_node(node, '\':\'', 1)
            node.add_child(q)
            node.add_child(lhs)
        elif node.symbol == 'var':
            is_reuse = self.get_inh_attr(inherit_atts, 'is_reuse')
            if is_reuse:
                candidates = list(self.defined_vars)
            else:
                candidates = list(self.full_var_set - self.defined_vars)            
            assert len(candidates)
            var_id = self.rand_att(node, candidates)
            
            c = self.get_node(node, '\'v\'', 0)
            node.add_child(c)

            i = self.get_node(node, 'var_id', 1)
            node.add_child(i)
            self.tree_generator(i, {'id' : var_id})

            if not is_reuse: # create a new lhs
                self.defined_vars.add(var_id)
        elif node.symbol == 'var_id':
            idx = self.get_inh_attr(inherit_atts, 'id')
            i = self.get_node(node, '\'%d\'' % idx, 0)
            node.add_child(i)
        else:
            assert node.symbol in ['program', 'lhs', 'rhs', 'expr', 'unary_expr', 'binary_expr', 'unary_op', 'unary_func', 'binary_op', 'binary_op', 'operand', 'immediate_number', 'digit']            
            rule = self.rand_rule(node)
            p = prod[node.symbol][rule]

            for i in range(len(p)):
                c = self.get_node(node, p[i], i)
                if not p[i][0] == '\'': # non-terminal
                    t = self.tree_generator(c, inherit_atts=inherit_atts)                    
                node.add_child(c)

    def decode(self, node, walker):
        self.walker = walker
        self.walker.reset()
        self.reset_state()
        self.tree_generator(node)

def batch_make_att_masks(node_list, tree_decoder = None, walker = None, dtype=np.byte):
    if walker is None:
        walker = OnehotBuilder()
    if tree_decoder is None:
        tree_decoder = ProgramOnehotBuilder()

    true_binary = np.zeros((len(node_list), cmd_args.max_decode_steps, DECISION_DIM), dtype=dtype)
    rule_masks = np.zeros((len(node_list), cmd_args.max_decode_steps, DECISION_DIM), dtype=dtype)

    for i in range(len(node_list)):
        node = node_list[i]
        tree_decoder.decode(node, walker)

        true_binary[i, np.arange(walker.num_steps), walker.global_rule_used[:walker.num_steps]] = 1
        true_binary[i, np.arange(walker.num_steps, cmd_args.max_decode_steps), -1] = 1

        for j in range(walker.num_steps):
            rule_masks[i, j, walker.mask_list[j]] = 1

        rule_masks[i, np.arange(walker.num_steps, cmd_args.max_decode_steps), -1] = 1.0

    return true_binary, rule_masks

if __name__ == '__main__':
    dec = ProgTreeDecoder()


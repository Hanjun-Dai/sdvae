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
from prog_util import prod, MAX_VARS, rule_ranges, MAX_NUM_STATEMENTS, TOTAL_NUM_RULES, DECISION_DIM

class DecodingLimitExceeded(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return 'DecodingLimitExceeded'

class ProgramOnehotBuilder(object):

    def __init__(self):
        super(ProgramOnehotBuilder, self).__init__()
        self.reset()
        
    def reset(self):
        self.num_steps = 0
        self.global_rule_used = []
        self.mask_list = []
    
    def sample_index_with_mask(self, node, idxes):
        assert node.rule_used is not None
        g_range = rule_ranges[node.symbol]
        global_idx = g_range[0] + node.rule_used
        self.global_rule_used.append(global_idx)
        self.mask_list.append(np.array(idxes))

        self.num_steps += 1

        result = None
        for i in range(len(idxes)):
            if idxes[i] == global_idx:
                result = i
        if result is None:
            print(node.symbol, idxes, global_idx)
        assert result is not None
        return result

    def sample_att(self, node, candidates):
        assert hasattr(node, 'var_id')
        assert node.var_id in candidates

        global_idx = TOTAL_NUM_RULES + node.var_id
        self.global_rule_used.append(global_idx)
        self.mask_list.append(np.array(candidates) + TOTAL_NUM_RULES)

        self.num_steps += 1
        
        return node.var_id
    
class ConditionalProgramDecoder(object):

    def __init__(self, raw_logits, use_random):
        super(ConditionalProgramDecoder, self).__init__()
        self.raw_logits = raw_logits
        self.use_random = use_random
        assert len(raw_logits.shape) == 2 and raw_logits.shape[1] == DECISION_DIM

        self.reset()

    def reset(self):
        self.num_steps = 0

    def _get_idx(self, cur_logits):
        if self.use_random:
            cur_prob = np.exp(cur_logits)
            cur_prob = cur_prob / np.sum(cur_prob)

            result = np.random.choice(len(cur_prob), 1, p=cur_prob)[0]
            result = int(result)  # enusre it's converted to int
        else:
            result = np.argmax(cur_logits)

        self.num_steps += 1
        return result

    def sample_index_with_mask(self, node, idxes):
        if self.num_steps >= self.raw_logits.shape[0]:
            raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[self.num_steps][idxes]
        
        return self._get_idx(cur_logits)

    def sample_att(self, node, candidates):        
        if self.num_steps >= self.raw_logits.shape[0]:
            raise DecodingLimitExceeded()
        cur_logits = self.raw_logits[self.num_steps][np.array(candidates) + TOTAL_NUM_RULES]

        idx = self._get_idx(cur_logits)
        
        return candidates[idx]

class PurelyRandomProgramDecoder(object):

    def __init__(self):
        super(PurelyRandomProgramDecoder, self).__init__()
        self.reset()

    def reset(self):
        pass

    def sample_in_candidates(self, raw_p, idxes):
        sub_p = np.array(raw_p)[ np.array(idxes) ]        
        sub_p /= np.sum(sub_p)        
        idx = np.random.choice(len(sub_p), 1, p=sub_p)[0]
        idx = int(idx)
        return idx

    def sample_index_with_mask(self, node, idxes):
        assert len(idxes)
        
        non_terminal = node.symbol
        p = [1]
        if non_terminal == 'stat_list':
            p = [0.5, 0.5]
        elif non_terminal == 'stat':
            p = [0.8, 0.2]
        elif non_terminal == 'expr':
            p = [0.2, 0.8]
        elif non_terminal == 'unary_expr':
            p = [0.5, 0.5]
        elif non_terminal == 'unary_func':
            p = [0.3, 0.3, 0.4]
        elif non_terminal == 'binary_op':
            p = [0.3, 0.3, 0.3, 0.1]
        elif non_terminal == 'operand':
            p = [0.5, 0.5]
        elif non_terminal == 'digit':
            p = [0.2] + [0.1] * 8

        if len(p) != len(prod[non_terminal]):
            print(non_terminal, len(p), len(prod[non_terminal]))
        assert len(p) == len(prod[non_terminal])
        assert len(p) >= len(idxes)        
        if len(p) > len(idxes):
            g_range = rule_ranges[non_terminal]
            result = self.sample_in_candidates(p, np.array( idxes ) - g_range[0])
        else:
            result = np.random.choice(len(p), 1, p=p)[0]
            
        result = int(result)  # enusre it's converted to int
        assert result < len(idxes)
        return result

    def sample_att(self, node, candidates):
        p = [0.1] * 10
        assert len(p) >= len(candidates)
        idx = self.sample_in_candidates(p, candidates)
        assert idx < len(candidates)
        return candidates[idx]

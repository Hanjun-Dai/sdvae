#!/usr/bin/env python

from collections import defaultdict
from cmd_args import cmd_args

info_folder = '../../dropbox/context_free_grammars'

prod = defaultdict(list)

_total_num_rules = 0
rule_ranges = {}
terminal_idxes = {}

with open(cmd_args.grammar_file, 'r') as f:
    for row in f:
        s = row.split('->')[0].strip()
        rules = row.split('->')[1].strip().split('|')
        rules = [w.strip() for w in rules]
        for rule in rules:
            rr = rule.split()
            prod[s].append(rr)
            for x in rr:
                if x[0] == '\'' and not x in terminal_idxes:
                    idx = len(terminal_idxes)
                    terminal_idxes[x] = idx
        rule_ranges[s] = (_total_num_rules, _total_num_rules + len(rules))
        _total_num_rules += len(rules)

TOTAL_NUM_RULES = _total_num_rules
MAX_VARS = 10
MAX_NUM_STATEMENTS = 10
DECISION_DIM = MAX_VARS + TOTAL_NUM_RULES + 2

if __name__ == '__main__':
    print(terminal_idxes)
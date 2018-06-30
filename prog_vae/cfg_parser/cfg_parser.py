#!/usr/bin/env python2

import random
import re

import nltk
from nltk.grammar import Nonterminal, Production


class Grammar(object):
    def __init__(self, filepath=None):
        if filepath:
            self.load(filepath)

    def load(self, filepath):
        cfg_string = ''.join(list(open(filepath).readlines()))

        # parse from nltk
        cfg_grammar = nltk.CFG.fromstring(cfg_string)
        # self.cfg_parser = cfg_parser = nltk.RecursiveDescentParser(cfg_grammar)
        self.cfg_parser = cfg_parser = nltk.ChartParser(cfg_grammar)

        # our info for rule macthing
        self.head_to_rules = head_to_rules = {}
        self.valid_tokens = valid_tokens = set()
        rule_ranges = {}
        total_num_rules = 0
        first_head = None
        for line in cfg_string.split('\n'):
            if len(line.strip()) > 0:
                head, rules = line.split('->')
                head = Nonterminal(head.strip())    # remove space
                rules = [_.strip() for _ in rules.split('|')]    # split and remove space
                rules = [
                    tuple([Nonterminal(_) if not _.startswith("'") else _[1:-1] for _ in rule.split()])
                    for rule in rules
                ]
                head_to_rules[head] = rules

                for rule in rules:
                    for t in rule:
                        if isinstance(t, str):
                            valid_tokens.add(t)

                if first_head is None:
                    first_head = head

                rule_ranges[head] = (total_num_rules, total_num_rules + len(rules))
                total_num_rules += len(rules)

        self.first_head = first_head

        self.rule_ranges = rule_ranges
        self.total_num_rules = total_num_rules

    def generate(self):
        frontier = [self.first_head]
        while True:
            is_ended = not any(isinstance(item, Nonterminal) for item in frontier)
            if is_ended:
                break
            for i in range(len(frontier)):
                item = frontier[i]
                if isinstance(item, Nonterminal):
                    replacement_id = random.randint(0, len(self.head_to_rules[item]) - 1)
                    replacement = self.head_to_rules[item][replacement_id]
                    frontier = frontier[:i] + list(replacement) + frontier[i + 1:]
                    break
        return ''.join(frontier)

    def tokenize(self, sent):
        s = sent
        funcs = ['sin', 'cos', 'exp']
        for fn in funcs:
            s = s.replace(fn + '(', fn + ' ')
        s = re.sub(r'([^a-z ])', r' \1 ', s)
        for fn in funcs:
            s = s.replace(fn, fn + ' (')
        return s.split()


class AnnotatedTree(object):
    '''Annotated Tree.

    It uses Nonterminal / Production class from nltk,
    see http://www.nltk.org/_modules/nltk/grammar.html for code.

    Attributes:
        symbol: a str object (for erminal) or a Nonterminal object (for non-terminal).
        children: a (maybe-empty) list of children.
        rule: a Production object.
        rule_selection_id: the 0-based index of which part of rule being selected. -1 for terminal.

    Method:
        is_leaf(): True iff len(children) == 0
    '''

    def __init__(self, symbol=None, children=None, rule=None, rule_selection_id=-1):
        symbol = symbol or ''
        children = children or []
        rule = rule or None
        # rule_selection_id = rule_selection_id or 0

        assert (len(children) > 0 and rule is not None) or (len(children) == 0 and rule is None)
        self.symbol = symbol
        self.children = children
        self.rule = rule
        self.rule_selection_id = rule_selection_id

    def is_leaf(self):
        return len(self.children) == 0

    def __str__(self):
        return '[Symbol = %s / Rule = %s / Rule Selection ID = %d / Children = %s]' % (
            self.symbol, self.rule, self.rule_selection_id, self.children
        )

    def __repr__(self):
        return self.__str__()


def parse(sent, grammar):
    '''Returns a list of trees
    (for it's possible to have multiple parse tree)

    Returns None if the parsing fails.
    '''
    # `sent` should be string
    assert isinstance(sent, str)

    sent = grammar.tokenize(sent)
    if sent is None:
        return None

    try:
        trees = list(grammar.cfg_parser.parse(sent))
    except ValueError:
        return None
    # print(trees)

    def _child_names(tree):
        names = []
        for child in tree:
            if isinstance(child, nltk.tree.Tree):
                names.append(Nonterminal(child._label))
            else:
                names.append(child)
        return names

    def _find_rule_selection_id(production):
        lhs, rhs = production.lhs(), production.rhs()
        assert lhs in grammar.head_to_rules
        rules = grammar.head_to_rules[lhs]
        for index, rule in enumerate(rules):
            if rhs == rule:
                return index
        assert False
        return 0

    def convert(tree):
        # convert from ntlk.tree.Tree to our AnnotatedTree

        if isinstance(tree, nltk.tree.Tree):
            symbol = Nonterminal(tree.label())
            children = list(convert(_) for _ in tree)
            rule = Production(Nonterminal(tree.label()), _child_names(tree))
            rule_selection_id = _find_rule_selection_id(rule)
            return AnnotatedTree(symbol=symbol, children=children, rule=rule, rule_selection_id=rule_selection_id)
        else:
            return AnnotatedTree(symbol=tree)

    trees = [convert(tree) for tree in trees]
    return trees

#!/usr/bin/env python3

import cfg_parser as parser


info_folder = '../../dropbox/context_free_grammars'
grammar = parser.Grammar(info_folder + '/mol_zinc.grammar')
ts = parser.parse('ClI=I=S(CBI)(-CN(C-N(N-C-F))I(S-I)C-C=I)', grammar)
t = ts[0]

print('(ugly) tree:')
print(t)
print()


print('for root:')
print('symbol is %s, is it non-terminal = %s, it\' value is %s (of type %s)' % (
    t.symbol,
    isinstance(t, parser.Nonterminal),
    t.symbol.symbol(),
    type(t.symbol.symbol())
))
print('rule is %s, its left side is %s (of type %s), its right side is %s, a tuple '
'which each element can be either str (for terminal) or Nonterminal (for nonterminal)' % (
   t.rule,
   t.rule.lhs(),
   type(t.rule.lhs()),
   t.rule.rhs(),
))

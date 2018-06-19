#!/bin/bash

grammar_file=../../dropbox/context_free_grammars/mol_zinc.grammar

python mol_encoder.py \
    -grammar_file $grammar_file \
    -mode gpu

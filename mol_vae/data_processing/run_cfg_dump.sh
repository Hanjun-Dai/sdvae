#!/bin/bash

dropbox=../../dropbox

grammar_file=$dropbox/context_free_grammars/mol_zinc.grammar
smiles_file=$dropbox/data/zinc/250k_rndm_zinc_drugs_clean.smi

python dump_cfg_trees.py \
    -grammar_file $grammar_file \
    -smiles_file $smiles_file

#!/bin/bash

dropbox=../../dropbox

sk=0
grammar_file=$dropbox/context_free_grammars/mol_zinc.grammar
smiles_file=$dropbox/data/zinc/250k_rndm_zinc_drugs_clean.smi

save_dir=$dropbox/data/zinc

python make_dataset_parallel.py \
    -grammar_file $grammar_file \
    -smiles_file $smiles_file \
    -save_dir $save_dir \
    -skip_deter $sk \
    -bondcompact 0 \
    -data_gen_threads 8

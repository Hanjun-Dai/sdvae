#!/bin/bash

dropbox=../../dropbox

grammar_file=$dropbox/context_free_grammars/prog_leftskew.grammar
save_dir=$dropbox/data/program-free_var_id-check_data
min_len=1
max_len=5
phase=train

python make_dataset_parallel.py \
    -grammar_file $grammar_file \
    -save_dir $save_dir \
    -min_len $min_len \
    -max_len $max_len \
    -phase $phase \
    -data_gen_threads 8

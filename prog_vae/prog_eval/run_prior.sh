#!/bin/bash

dropbox=../../dropbox

grammar_file=$dropbox/context_free_grammars/prog.grammar

min_len=1
max_len=5
data_dump=$dropbox/data/program/nbstate-${min_len}-to-${max_len}.h5

bsize=350
enc=cnn
ae_type=vae
loss_type=perplexity
rnn_type=gru
hidden=200
dense=200
c1=7
c2=8
c3=9
kl_coeff=1
lr=0.001
num_epochs=1000
eps_std=0.01
save_dir=$HOME/scratch/results/graph_generation/prog_${ae_type}/enc-${enc}-loss-${loss_type}-h-${hidden}-d-${dense}-c-${c1}${c2}${c3}-eps-${eps_std}-rnn-${rnn_type}-kl-${kl_coeff}

export CUDA_VISIBLE_DEVICES=0
python valid_prior.py \
    -grammar_file $grammar_file \
    -data_dump $data_dump \
    -batch_size $bsize \
    -encoder_type $enc \
    -hidden $hidden \
    -dense $dense \
    -c1 $c1 \
    -c2 $c2 \
    -c3 $c3 \
    -save_dir $save_dir \
    -ae_type $ae_type \
    -learning_rate $lr \
    -rnn_type $rnn_type \
    -num_epochs $num_epochs \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -kl_coeff $kl_coeff \
    -mode gpu \
    -saved_model $save_dir/epoch-best.model

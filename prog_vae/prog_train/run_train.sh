#!/bin/bash

dropbox=../../dropbox

grammar_file=$dropbox/context_free_grammars/prog_leftskew.grammar

dtype=free_var_id-check_data
min_len=1
max_len=5
data_dump=$dropbox/data/program-${dtype}/nbstate-${min_len}-to-${max_len}-train.h5

bsize=500
enc=cnn
ae_type=vae
loss_type=perplexity
rnn_type=sru
latent=56
hidden=200
dense=200
c1=7
c2=8
c3=9
kl_coeff=0.015
lr=0.001
num_epochs=1000
eps_std=0.01
save_dir=$HOME/scratch/results/graph_generation/prog-${dtype}-${ae_type}/enc-${enc}-loss-${loss_type}-latent-${latent}-h-${hidden}-d-${dense}-c-${c1}${c2}${c3}-eps-${eps_std}-rnn-${rnn_type}-kl-${kl_coeff}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0
python train_prog.py \
    -grammar_file $grammar_file \
    -data_dump $data_dump \
    -batch_size $bsize \
    -encoder_type $enc \
    -latent_dim $latent \
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
    #-saved_model $save_dir/iter-100.model

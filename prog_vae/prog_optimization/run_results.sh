#!/bin/bash

dropbox=../../dropbox

grammar_file=$dropbox/context_free_grammars/prog_leftskew.grammar

min_len=1
max_len=5
prefix=free_var_id-check_data

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
eps_std=0.01
save_dir=$dropbox/sdvae/prog-${prefix}-${ae_type}/enc-${enc}-loss-${loss_type}-latent-${latent}-h-${hidden}-d-${dense}-c-${c1}${c2}${c3}-eps-${eps_std}-rnn-${rnn_type}-kl-${kl_coeff}

model=epoch-best
lr=0.0005
out_dir=$save_dir/bo-$model

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

python get_results.py \
    -save_dir $out_dir \
    -y_norm 1 \
    -prog_idx $1

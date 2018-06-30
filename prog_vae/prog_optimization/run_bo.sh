#!/bin/bash

dropbox=../../dropbox

grammar_file=$dropbox/context_free_grammars/prog_leftskew.grammar

min_len=1
max_len=5
prefix=free_var_id-check_data

enc=cnn
ae_type=vae
loss_type=perplexity
rnn_type=gru
latent=56
hidden=200
dense=200
c1=7
c2=8
c3=9
kl_coeff=0.015
eps_std=0.01
save_dir=$HOME/scratch/results/graph_generation/prog-${prefix}-${ae_type}/enc-${enc}-loss-${loss_type}-latent-${latent}-h-${hidden}-d-${dense}-c-${c1}${c2}${c3}-eps-${eps_std}-rnn-${rnn_type}-kl-${kl_coeff}

model=epoch-best
prog_idx=0
y_norm=0
seed=1
num_epochs=40
lr=0.0005
out_dir=$save_dir/bo-$model

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

python bayesian_optimization.py \
    -grammar_file $grammar_file \
    -encoder_type $enc \
    -latent_dim $latent \
    -prog_idx $prog_idx \
    -phase train \
    -data_dir $dropbox/data/program-$prefix \
    -dense $dense \
    -min_len $min_len \
    -y_norm $y_norm \
    -max_len $max_len \
    -prefix $prefix \
    -c1 $c1 \
    -c2 $c2 \
    -c3 $c3 \
    -save_dir $out_dir \
    -ae_type $ae_type \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -gp_lr $lr \
    -seed $seed \
    -feature_dump $save_dir/featuredump-$model/train-${min_len}-${max_len}-features.npy \
    -num_epochs $num_epochs \
    -rnn_type $rnn_type \
    -mode gpu \
    -saved_model $save_dir/${model}.model

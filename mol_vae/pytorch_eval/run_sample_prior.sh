#!/bin/bash

dropbox=../../dropbox

grammar_file=$dropbox/context_free_grammars/mol_zinc.grammar

bsize=500
enc=cnn
ae_type=vae
loss_type=vanilla
rnn_type=gru
kl_coeff=1
lr=0.001
num_epochs=255
eps_std=0.01
sk=0
seed=100
save_dir=$dropbox/results/zinc

echo "save_dir for use is $save_dir"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # only set when CUDA_VISIBLE_DEVICES is not set.
python sample_prior.py \
    -grammar_file $grammar_file \
    -batch_size $bsize \
    -seed $seed \
    -skip_deter 0 \
    -old 0 \
    -encoder_type $enc \
    -save_dir $save_dir \
    -ae_type $ae_type \
    -learning_rate $lr \
    -rnn_type $rnn_type \
    -num_epochs $num_epochs \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -kl_coeff $kl_coeff \
    -mode gpu \
    -saved_model $save_dir/zinc_kl_sum.model \
    $@ ;

python eval_similarity.py $save_dir/zinc_kl_sum.model-sampled_prior.txt

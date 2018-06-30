#!/bin/bash

dropbox=../../dropbox

grammar_file=$dropbox/context_free_grammars/mol_zinc.grammar
smiles_file=$dropbox/data/zinc/250k_rndm_zinc_drugs_clean.smi

enc=cnn
ae_type=vae
loss_type=vanilla
eps_std=0.01
rnn_type=gru
kl=1
sk=0

save_dir=$dropbox/results/zinc

model=zinc_kl_sum
noisy=0
round=0
out_dir=$save_dir/featuredump-$model

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # only set when CUDA_VISIBLE_DEVICES is not set.
python dump_encoding.py \
    -grammar_file $grammar_file \
    -smiles_file $smiles_file \
    -batch_size 10000 \
    -encoder_type $enc \
    -save_dir $out_dir \
    -skip_deter $sk \
    -old 0 \
    -ae_type $ae_type \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -rnn_type $rnn_type \
    -kl_coeff $kl \
    -noisy $noisy \
    -round $round \
    -mode gpu \
    -saved_model $save_dir/${model}.model \
    $@ ;

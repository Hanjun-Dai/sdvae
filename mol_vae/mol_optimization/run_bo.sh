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
seed=1
num_epochs=50
lr=0.0005
out_dir=$save_dir/bo-$model

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

python bayesian_optimization.py \
    -grammar_file $grammar_file \
    -smiles_file $smiles_file \
    -encoder_type $enc \
    -save_dir $out_dir \
    -ae_type $ae_type \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -skip_deter $sk \
    -gp_lr $lr \
    -seed $seed \
    -feature_dump $save_dir/featuredump/${model}-features.npy \
    -zinc_dir $dropbox/data/zinc \
    -num_epochs $num_epochs \
    -rnn_type $rnn_type \
    -kl_coeff $kl \
    -mode gpu \
    -saved_model $save_dir/${model}.model

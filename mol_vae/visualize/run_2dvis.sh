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
sample_idx=4
axis1=0
axis2=1
gap=0.7
grid=13
proj_type=proj

out_dir=$save_dir/2dvis-$model

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

python draw_2dvis.py \
    -grammar_file $grammar_file \
    -smiles_file $smiles_file \
    -encoder_type $enc \
    -save_dir $out_dir \
    -skip_deter $sk \
    -old 0 \
    -ae_type $ae_type \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -rnn_type $rnn_type \
    -kl_coeff $kl \
    -grid_size $grid \
    -gap $gap \
    -axisone $axis1 \
    -axistwo $axis2 \
    -sample_idx $sample_idx \
    -mode gpu \
    -proj_type $proj_type \
    -saved_model $save_dir/${model}.model

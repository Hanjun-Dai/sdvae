#!/bin/bash

set -e

dropbox=../../dropbox

save_dir=$dropbox/data/program-free_var_id-check_data/

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

declare -a size_list=(50000)
#declare -a nbstat_list=(1 2 3 4 5 6 7 8 9 10)
declare -a nbstat_list=(1 2 3 4 5)
gold_prog_file=gold_prog.txt

for size in "${size_list[@]}" ; do
  for nbstat in "${nbstat_list[@]}" ; do
    main_name=free_var_id-check_data-number-$size-nbstat-$nbstat
    save_file=$save_dir/$main_name.txt
    save_test_file=$save_dir/$main_name.test.txt

    python precompute_bo_target.py \
      --prog_file $save_file \
      --gold_prog_file "${gold_prog_file}" \
      ;
    python precompute_bo_target.py \
      --prog_file $save_test_file \
      --gold_prog_file "${gold_prog_file}" \
      ;
  done
done

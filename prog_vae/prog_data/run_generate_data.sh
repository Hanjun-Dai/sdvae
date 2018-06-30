#!/bin/bash

set -e

dropbox=../../dropbox

save_dir=$dropbox/data/program-free_var_id-check_data/

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

declare -a size_list=(50000)
declare -a nbstat_list=(1 2 3 4 5)

for size in "${size_list[@]}" ; do
  for nbstat in "${nbstat_list[@]}" ; do
    main_name=free_var_id-check_data-number-$size-nbstat-$nbstat
    save_file=$save_dir/$main_name.txt
    save_test_file=$save_dir/$main_name.test.txt
    rm -rf $save_file
    rm -rf $save_test_file
    echo "generating $save_file (and $save_test_file for testing)"
    python generate_data.py \
      --number $size \
      --nb_stat $nbstat \
      --save_file $save_file \
      --save_test_file $save_test_file \
      --check_data=true \
      --free_var_id=true \
      &
  done
done


echo "Waiting for all jobs..."
wait
echo "Done."

#!/bin/bash

names=("t_25_change_reticulation_prob", "t_25_change_reticulation_count", "t_25_change_brlen_scaler", "t_25_unpartitioned")

for name in ${names[@]}
do
    sbatch -N 1 -n 1 submit_experiments_haswell.sh $name $name 1
done
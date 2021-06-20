#!/bin/bash

declare -a names=(
    "t_20_change_reticulation_prob"
    "t_20_change_reticulation_count"
    "t_20_change_brlen_scaler"
    "t_20_unpartitioned"
    )

for name in ${names[*]}
do
    sbatch -N 1 -n 1 submit_experiments_haswell.sh "$name" "$name" 1
done
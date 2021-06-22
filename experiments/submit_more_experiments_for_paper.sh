#!/bin/bash

declare -a n_taxa_list=(
    "10"
    "15"
    "20"
    "25"
    "30"
    "35"
    "40"
)

declare -a n_reticulations_list=(
    "1"
    "2"
    "3"
    "4"
)

declare -a exp_name_list=(
    "standard_random"
    "brlen_linkage"
)

for n_taxa in ${n_taxa_list[*]}
do
    for n_reticulations in ${n_reticulations_list[*]}
    do
        for exp_name in ${exp_name_list[*]}
        do
            name="t_"$n_taxa"_r_"$n_reticulations"_"$exp_name
            sbatch -N 1 -n 1 submit_experiments_haswell.sh "$name" "$name" 1
        done
    done
done
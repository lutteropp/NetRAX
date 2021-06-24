#!/bin/bash

n_taxa="40"
n_reticulations="4"
n_runs=100
exp_name="standard_multi"

for (( c=1; c<=$n_runs; c++ ))
do  
   name="t_"$n_taxa"_r_"$n_reticulations"_"$exp_name
   sbatch -N 1 -n 1 submit_experiments_haswell.sh "$name" "$name" 1
done

for n_taxa in ${n_taxa_list[*]}
do
    for n_reticulations in ${n_reticulations_list[*]}
    do
        for exp_name in ${exp_name_list[*]}
        do
            name="t_"$n_taxa"_r_"$n_reticulations"_"$exp_name
            sbatch -N 1 -n 1 submit_experiments_haswell.sh "$name" "$name"_multi_"$c" 1
        done
    done
done
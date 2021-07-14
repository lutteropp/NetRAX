#!/bin/bash

for i in {0..49}
do
    name="A_10_1_run_$i"
    sbatch -N 1 -n 1 submit_experiments_haswell.sh "t_10_r_1_standard_random" "$name" 1

    name="A_20_2_run_$i"
    sbatch -N 1 -n 1 submit_experiments_haswell.sh "t_20_r_2_standard_random" "$name" 1

    name="A_30_3_run_$i"
    sbatch -N 1 -n 1 submit_experiments_haswell.sh "t_30_r_3_standard_random" "$name" 1

    name="B_20_1_run_$i"
    sbatch -N 1 -n 1 submit_experiments_haswell.sh "t_20_r_1_change_reticulation_prob" "$name" 1

    name="C_20_1_run_$i"
    sbatch -N 1 -n 1 submit_experiments_haswell.sh "t_20_r_1_unpartitioned" "$name" 1
done
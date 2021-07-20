#!/bin/bash

name="D_scramble"
echo $name
sbatch -N 1 -n 1 submit_scramble_partitions_experiment_haswell.sh

name="E_psize"
echo $name
sbatch -N 1 -n 1 submit_experiments_haswell.sh "t_30_r_3_partition_size" "$name" 1
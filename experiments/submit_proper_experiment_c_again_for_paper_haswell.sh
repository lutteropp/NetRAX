#!/bin/bash

for i in {0..49}
do
    name="C_20_2_run_$i"
    echo $name
    sbatch -N 1 -n 1 submit_experiments_haswell.sh "t_20_r_2_unpartitioned" "$name" 1
done

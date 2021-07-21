#!/bin/bash

# run the proper, non-anecdotal experiments

for i in {0..49}
do
    name="A_10_1_run_$i"
    echo $name
    bash submit_experiments_phd_laptop.sh "t_10_r_1_standard_random" "$name" 1
done

for i in {0..49}
do
    name="A_20_2_run_$i"
    echo $name
    bash submit_experiments_phd_laptop.sh "t_20_r_2_standard_random" "$name" 1
done

for i in {0..49}
do
    name="B_20_1_run_$i"
    echo $name
    bash submit_experiments_phd_laptop.sh "t_20_r_1_change_reticulation_prob" "$name" 1
done

for i in {0..49}
do
    name="C_20_2_run_$i"
    echo $name
    bash submit_experiments_phd_laptop.sh "t_20_r_2_unpartitioned" "$name" 1
done

for i in {0..49}
do
    name="A_30_3_run_$i"
    echo $name
    bash submit_experiments_phd_laptop.sh "t_30_r_3_standard_random" "$name" 1
done

for i in {0..49}
do
    name="A_40_4_run_$i"
    echo $name
    bash submit_experiments_phd_laptop.sh "t_40_r_4_standard_random" "$name" 1
done
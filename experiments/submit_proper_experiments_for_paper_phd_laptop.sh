#!/bin/bash

# run the proper, non-anecdotal experiments
sh submit_experiments_phd_laptop.sh "t_10_r_1_standard_random" 50
sh submit_experiments_phd_laptop.sh "t_20_r_2_standard_random" 50
sh submit_experiments_phd_laptop.sh "t_30_r_3_standard_random" 50
sh submit_experiments_phd_laptop.sh "t_40_r_4_standard_random" 50
sh submit_experiments_phd_laptop.sh "t_20_r_1_change_reticulation_prob" 50
sh submit_experiments_phd_laptop.sh "t_20_r_1_unpartitioned" 50
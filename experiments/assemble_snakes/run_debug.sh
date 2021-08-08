#!/bin/bash

root=/home/luttersh/NetRAX/experiments/assemble_snakes
netrax=/home/luttersh/NetRAX/bin/netrax
raxml=/home/luttersh/NetRAX/experiments/deps/raxml-ng
ali_for_raxml=$root/snakes_msa.fasta
ali=$root/assemble_snakes.raxml.rba
model=$root/snakes_partitions.txt
raxtree=$root/assemble_snakes.raxml.bestTree
outdir=$root

time mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/snakes_random_best_inferred_network_debug.nw --best_displayed_tree_variant --brlen linked --num_parsimony_start_networks 1 --num_random_start_networks 0 | tee netrax_snakes_random_best_debug_output.txt

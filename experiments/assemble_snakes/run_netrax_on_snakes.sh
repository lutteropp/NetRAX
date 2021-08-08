#!/bin/bash

root=/home/sarah/eclipse-workspace/NetRAX/experiments/assemble_snakes
netrax=/home/sarah/eclipse-workspace/NetRAX/bin/netrax
raxml=/home/sarah/eclipse-workspace/NetRAX/experiments/deps/raxml-ng
ali_for_raxml=$root/snakes_msa.fasta
ali=$root/assemble_snakes.raxml.rba
model=$root/snakes_partitions.txt
raxtree=$root/assemble_snakes.raxml.bestTree
outdir=$root

#time $raxml --msa $ali_for_raxml --model $model -seed 42 --prefix $outdir --redo
time mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/snakes_best_inferred_network.nw --best_displayed_tree_variant --brlen linked --start_network $raxtree --good_start
time mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/snakes_average_inferred_network.nw --average_displayed_tree_variant --brlen linked --start_network $raxtree --good_start

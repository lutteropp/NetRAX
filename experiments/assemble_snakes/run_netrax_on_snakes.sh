#!/bin/bash

root=/home/luttersh/NetRAX/experiments/assemble_snakes
netrax=/home/luttersh/NetRAX/bin/netrax
raxml=/home/luttersh/NetRAX/experiments/deps/raxml-ng
ali_for_raxml=$root/snakes_msa.fasta
ali=$root/assemble_snakes.raxml.rba
model=$root/snakes_partitions.txt
raxtree=$root/assemble_snakes.raxml.bestTree
outdir=$root

time $raxml --msa $ali_for_raxml --model $model -seed 42 --prefix $outdir/assemble_snakes --redo | tee raxml_snakes_output.txt > snakes_raxml_time.txt

time mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/snakes_ml_best_inferred_network.nw --best_displayed_tree_variant --brlen linked --start_network $raxtree --good_start | tee netrax_snakes_ml_best_output.txt > snakes_ml_best_time.txt

time mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/snakes_ml_average_inferred_network.nw --average_displayed_tree_variant --brlen linked --start_network $raxtree --good_start | tee netrax_snakes_ml_average_output.txt > snakes_ml_average_time.txt

time mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/snakes_random_best_inferred_network.nw --best_displayed_tree_variant --brlen linked -p 3 -n 3 | tee netrax_snakes_random_best_output.txt > snakes_random_best_time.txt

time mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/snakes_random_average_inferred_network.nw --average_displayed_tree_variant --brlen linked -p 3 -n 3 | tee netrax_snakes_random_average_output.txt > snakes_random_average_time.txt
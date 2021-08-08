#!/bin/bash

netraxfolder=/home/luttersh/NetRAX
#netraxfolder=/home/sarah/code-workspace/NetRAX

build_start_trees_script=$netraxfolder/build_start_trees.py
netrax_script=$netraxfolder/netrax.py

root=$netraxfolder/experiments/assemble_snakes
ali_for_raxml=$root/snakes_msa.fasta
ali=$root/snakes.raxml.rba
model=$root/snakes_partitions.txt
besttree=$root/snakes.raxml.bestTree
mltrees=$root/snakes.raxml.mlTrees
mltrees_unique=$root/snakes.raxml.mlTrees.unique
badtrees=$root/snakes.raxml.startTrees
badtrees_unique=$root/snakes.raxml.startTrees.unique
outdir=$root

python3 ${build_start_trees_script} --msa_path ${ali_for_raxml} --partitions_path $model --seed 42 --name snakes --num_parsimony_trees 10 --num_random_trees 10

python3 ${netrax_script} --name "snakes_single_best" --msa_path $ali --partitions_path $model --likelihood_type best --brlen_linkage_type linked --seed 42 --start_networks $besttree --good_start
python3 ${netrax_script} --name "snakes_single_average" --msa_path $ali --partitions_path $model --likelihood_type average --brlen_linkage_type linked --seed 42 --start_networks $besttree --good_start
python3 ${netrax_script} --name "snakes_multi_best" --msa_path $ali --partitions_path $model --likelihood_type best --brlen_linkage_type linked --seed 42 --start_networks $mltrees_unique --good_start
python3 ${netrax_script} --name "snakes_multi_average" --msa_path $ali --partitions_path $model --likelihood_type average --brlen_linkage_type linked --seed 42 --start_networks $mltrees_unique --good_start

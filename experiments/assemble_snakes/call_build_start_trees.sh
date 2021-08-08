#!/bin/bash

#netrax_folder=home/luttersh/NetRAX
netraxfolder=/home/sarah/code-workspace/NetRAX

build_start_trees_script=$netraxfolder/build_start_trees.py
multirun_script=$netraxfolder/netrax_multi.py

root=$netraxfolder/experiments/assemble_snakes
netrax=$netraxfolder/bin/netrax
ali_for_raxml=$root/snakes_msa.fasta
ali=$root/assemble_snakes.raxml.rba
model=$root/snakes_partitions.txt
besttree=$root/assemble_snakes.raxml.bestTree
mltrees=$root/assemble_snakes.raxml.mlTrees
mltrees_unique=$root/assemble_snakes.raxml.mlTrees_unique
badtrees=$root/assemble_snakes.raxml.startTrees
badtrees_unique=$root/assemble_snakes.raxml.startTrees_unique
outdir=$root

#this command also builds the other start tree types
#build all the RAxML-NG ML trees, keep only unique topologies
python3 ${build_start_trees_script} --msa_path $ali_for_raxml --partitions_path $model --seed 42 --start_trees_output_path ${mltrees_unique} --num_parsimony_trees 10 --num_random_trees 10 --keep_only_unique

#build random and parsimony trees, without inference, keep only unique topologies
#python3 ${build_start_trees_script} --msa_path $ali_for_raxml --partitions_path $model --seed 42 --start_trees_output_path ${badtrees_unique} --no_inference --num_parsimony_trees 10 --num_random_trees 10 --keep_only_unique

#build the RAxML-NG best tree
#python3 ${build_start_trees_script} --msa_path $ali_for_raxml --partitions_path $model --seed 42 --start_trees_output_path $besttree --take_only_best_tree --num_parsimony_trees 10 --num_random_trees 10

#build all the RAxML-NG ML trees
#python3 ${build_start_trees_script} --msa_path $ali_for_raxml --partitions_path $model --seed 42 --start_trees_output_path $mltrees --num_parsimony_trees 10 --num_random_trees 10

#build random and parsimony trees, without inference
#python3 ${build_start_trees_script} --msa_path $ali_for_raxml --partitions_path $model --seed 42 --start_trees_output_path $badtrees --no_inference --num_parsimony_trees 10 --num_random_trees 10
#!/bin/bash

netraxfolder=home/luttersh/NetRAX
#netraxfolder=/home/sarah/code-workspace/NetRAX

build_start_trees_script=$netraxfolder/build_start_trees.py

root=$netraxfolder/experiments/assemble_snakes
netrax=$netraxfolder/bin/netrax
ali_for_raxml=$root/snakes_msa.fasta
ali=$root/snakes.raxml.rba
model=$root/snakes_partitions.txt
besttree=$root/snakes.raxml.bestTree
mltrees=$root/snakes.raxml.mlTrees
mltrees_unique=$root/snakes.raxml.mlTrees.unique
badtrees=$root/snakes.raxml.startTrees
badtrees_unique=$root/snakes.raxml.startTrees.unique
outdir=$root

#this command also builds the other start tree types
python3 ${build_start_trees_script} --msa_path $ali_for_raxml --partitions_path $model --seed 42 --name snakes --num_parsimony_trees 10 --num_random_trees 10
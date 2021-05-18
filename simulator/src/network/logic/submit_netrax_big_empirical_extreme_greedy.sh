#!/bin/bash
#SBATCH -o netrax_%j.out
#SBATCH -B 2:8:1
#SBATCH --ntasks-per-node=1
##SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=compute_bound
#SBATCH -t 24:00:00
#SBATCH -N 32
#SBATCH -n 32
 
module purge
module load gompi/2019a

root=/home/luttersh/NetRAX/simulator/src/network/logic/data
netrax=/home/luttersh/NetRAX/bin/netrax
ali=$root/datasets_big_empirical/merged_genes_msa.txt.raxml.rba
model=$root/datasets_big_empirical/merged_genes_partitions.txt
raxtree=$root/datasets_big_empirical/merged_genes.raxml.bestTree
outdir=$root/datasets_big_empirical

mkdir -p $outdir

mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/merged_genes_inferred_network_greedy.nw --best_displayed_tree_variant --brlen linked --start_network $raxtree --no_rspr_moves

#!/bin/bash
#SBATCH -o netrax_%j.out
#SBATCH -B 2:8:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=compute_bound
#SBATCH -t 24:00:00
#SBATCH -N 64
#SBATCH -n 1024
 
module purge
module load gompi/2019a

root=/home/luttersh/NetRAX/experiments/data
netrax=/home/luttersh/NetRAX/bin/netrax
ali=$root/datasets_big_empirical/merged_genes_msa.txt.raxml.rba
model=$root/datasets_big_empirical/merged_genes_partitions.txt
raxtree=$root/datasets_big_empirical/merged_genes.raxml.bestTree
outdir=$root/datasets_big_empirical

mkdir -p $outdir

mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/merged_genes_inferred_network.nw --best_displayed_tree_variant --brlen linked --start_network $raxtree --good_start

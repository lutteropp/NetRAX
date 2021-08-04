#!/bin/bash
#SBATCH -o netrax_%j.out
#SBATCH -B 2:8:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=compute_bound
#SBATCH -t 24:00:00
#SBATCH -N 42
#SBATCH -n 672
 
module purge
module load gompi/2019a

root=/home/luttersh/NetRAX/experiments/subsampled_empirical
netrax=/home/luttersh/NetRAX/bin/netrax
ali=$root/merged_genes_species_msa.txt.raxml.rba
model=$root/merged_genes_species_partitions.txt
raxtree=$root/merged_genes_species.raxml.bestTree
#raxtree=$root/start_network.txt
outdir=$root

mkdir -p $outdir

mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/merged_genes_species_average_inferred_network.nw --average_displayed_tree_variant --brlen linked --start_network $raxtree --good_start
#!/bin/bash
#SBATCH -o netrax_%j.out
#SBATCH -B 2:8:1
#SBATCH --ntasks-per-node=1
##SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=compute_bound
#SBATCH -t 24:00:00
#SBATCH -N 30
#SBATCH -n 30
 
module purge
module load gompi/2019a

root=/home/luttersh/NetRAX/experiments/subsampled_empirical
netrax=/home/luttersh/NetRAX/bin/netrax
ali=$root/merged_genes_species_msa.txt.raxml.rba
model=$root/merged_genes_species_partitions.txt
raxtree=$root/start_network.txt
outdir=$root

mkdir -p $outdir

mpiexec $netrax --msa $ali --model $model --seed 42 --output $outdir/merged_genes_species_inferred_network.nw --best_displayed_tree_variant --brlen linked --start_network $raxtree --good_start

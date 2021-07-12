#!/bin/bash
#SBATCH -o raxng_%j.out
#SBATCH -B 2:8:1
#SBATCH --ntasks-per-node=1
##SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=compute_bound
#SBATCH -t 24:00:00
#SBATCH -N 10
#SBATCH -n 10
 
module purge
module load gompi/2019a

root=/home/luttersh/NetRAX/experiments/data
raxng=/home/luttersh/raxml-ng/bin/raxml-ng-mpi
ali=$root/datasets_big_empirical/merged_genes_species_msa.txt.raxml.rba
model=$root/datasets_big_empirical/merged_genes_species_partitions.txt
outdir=$root/datasets_big_empirical

mkdir -p $outdir

mpirun $raxng --search --msa $ali --seed 42 --prefix $outdir --redo --site-repeats off --workers 1

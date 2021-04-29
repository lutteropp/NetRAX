#!/bin/bash
#SBATCH -o raxng_%j.out
#SBATCH -B 2:8:1
#SBATCH --ntasks-per-node=1
##SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=compute_bound
#SBATCH -t 24:00:00
#SBATCH -N 20
#SBATCH -n 20
 
module purge
module load gompi/2019a

root=/home/luttersh/NetRAX/simulator/src/network/logic/data
raxng=/home/luttersh/NetRAX/libs/raxml-ng/bin/raxml-ng-mpi
ali=$root/datasets_big_empirical/merged_genes_msa.txt.raxml.rba
model=$root/datasets_big_empirical/merged_genes_partitions.txt
outdir=$root/datasets_big_empirical

mkdir -p $outdir

mpiexec $raxng --search --msa $ali --model $model --seed 42 --prefix $outdir --redo

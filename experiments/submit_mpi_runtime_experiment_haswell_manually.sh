#!/bin/sh

#source /etc/profile.d/modules.sh

#SBATCH -B 2:8:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -N 4
#SBATCH -n 4

module purge
module load Python
module load gompi/2019a
module load CMake/3.13.3-GCCcore-8.2.0
module load slurm

USAGE="Usage: sh submit_mpi_runtime_experiment_haswell_manually.sh ITERATION"

if [ $# -lt 1git  ]; then
    echo "Illegal number of parameters. ${USAGE}"
    exit 2
fi

ITERATION=$1

SCRIPTS="/home/luttersh/NetRAX/experiments/src"
FOLDER_PATH="/home/luttersh/NetRAX/experiments/data"
SUBFOLDER_PATH=${FOLDER_PATH}/${PREFIX}
LOGS_PATH=${FOLDER_PATH}/logs_${PREFIX}

python3 ${SCRIPTS}/mpi_runtime_experiment.py --iteration $ITERATION

/home/luttersh/NetRAX/experiments/deps/raxml-ng --msa data/datasets_mpi_runtime_$ITERATION/0_0_msa.txt --model data/datasets_mpi_runtime_$ITERATION/0_0_partitions.txt --prefix data/datasets_mpi_runtime_$ITERATION/0_0 --seed 42

echo "iteration,nprocs,likelihood_model,runtime_in_seconds" > mpi_experiment_results_$ITERATION.csv
for procs in {64,32,16,8,4,2,1}
do
    start_best=`date +%s`
    mpiexec -np $procs /home/luttersh/NetRAX/bin/netrax --msa data/datasets_mpi_runtime_$ITERATION/0_0_msa.txt --output data/datasets_mpi_runtime_$ITERATION/$procs_0_BEST_LINKED_FROM_RAXML_inferred_network.nw --model data/datasets_mpi_runtime_$ITERATION/0_0_partitions.txt --best_displayed_tree_variant --start_network data/datasets_mpi_runtime_$ITERATION/0_0.raxml.bestTree --good_start --brlen linked --seed 42
    end_best=`date +%s`
    runtime_best=`expr $end_best - $start_best`
    echo $iteration","$procs",BEST,"$runtime_best >> mpi_experiment_results_$ITERATION.csv

    start_avg=`date +%s`
    mpiexec -np $procs /home/luttersh/NetRAX/bin/netrax --msa data/datasets_mpi_runtime_$ITERATION/0_0_msa.txt --output data/datasets_mpi_runtime_$ITERATION/$procs_0_AVERAGE_LINKED_FROM_RAXML_inferred_network.nw --model data/datasets_mpi_runtime_$ITERATION/0_0_partitions.txt --average_displayed_tree_variant --start_network data/datasets_mpi_runtime_$ITERATION/0_0.raxml.bestTree --good_start --brlen linked --seed 42
    end_avg=`date +%s`
    runtime_avg=`expr $end_avg - $start_avg`
    echo $iteration","$procs",AVERAGE,"$runtime_avg >> mpi_experiment_results_$ITERATION.csv
done
#!/bin/sh

#source /etc/profile.d/modules.sh

#SBATCH -B 2:8:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -N 8
#SBATCH -n 128

module purge
module load Python
module load gompi/2019a
module load CMake/3.13.3-GCCcore-8.2.0
module load slurm

USAGE="Usage: sh submit_mpi_runtime_experiment_2_haswell_manually.sh ITERATION"

if [ $# -lt 1git  ]; then
    echo "Illegal number of parameters. ${USAGE}"
    exit 2
fi

ITERATION=$1

SCRIPTS="/home/luttersh/NetRAX/experiments/src"
FOLDER_PATH="/home/luttersh/NetRAX/experiments/data"
SUBFOLDER_PATH=${FOLDER_PATH}/${PREFIX}
LOGS_PATH=${FOLDER_PATH}/logs_${PREFIX}

python3 ${SCRIPTS}/mpi_runtime_experiment2.py --iteration ${ITERATION}

/home/luttersh/NetRAX/experiments/deps/raxml-ng --msa data/datasets_mpi_runtime2_${ITERATION}/0_0_msa.txt --model data/datasets_mpi_runtime2_${ITERATION}/0_0_partitions.txt --prefix data/datasets_mpi_runtime2_${ITERATION}/0_0 --seed 42

echo "iteration,n_procs,likelihood_type,runtime_in_seconds" > mpi_experiment2_results_${ITERATION}.csv
for procs in {128,112,96,80,64,48,32,16}
do
    start_best=`date +%s`
    mpiexec -np ${procs} /home/luttersh/NetRAX/bin/netrax --msa data/datasets_mpi_runtime2_${ITERATION}/0_0_msa.txt --output data/datasets_mpi_runtime2_${ITERATION}/${procs}_0_BEST_LINKED_FROM_RAXML_inferred_network.nw --model data/datasets_mpi_runtime2_${ITERATION}/0_0_partitions.txt --best_displayed_tree_variant --start_network data/datasets_mpi_runtime2_${ITERATION}/0_0.raxml.bestTree --good_start --brlen linked --seed 42
    end_best=`date +%s`
    runtime_best=`expr ${end_best} - ${start_best}`
    echo ${ITERATION}","${procs}",BEST,"$runtime_best >> mpi_experiment2_results_${ITERATION}.csv

    start_avg=`date +%s`
    mpiexec -np ${procs} /home/luttersh/NetRAX/bin/netrax --msa data/datasets_mpi_runtime2_${ITERATION}/0_0_msa.txt --output data/datasets_mpi_runtime2_${ITERATION}/${procs}_0_AVERAGE_LINKED_FROM_RAXML_inferred_network.nw --model data/datasets_mpi_runtime2_${ITERATION}/0_0_partitions.txt --average_displayed_tree_variant --start_network data/datasets_mpi_runtime2_${ITERATION}/0_0.raxml.bestTree --good_start --brlen linked --seed 42
    end_avg=`date +%s`
    runtime_avg=`expr ${end_avg} - ${start_avg}`
    echo ${ITERATION}","${procs}",AVERAGE,"$runtime_avg >> mpi_experiment2_results_${ITERATION}.csv
done
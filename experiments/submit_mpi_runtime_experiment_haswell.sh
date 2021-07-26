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

USAGE="Usage: sh submit_mpi_runtime_experiment_haswell.sh ITERATION"

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

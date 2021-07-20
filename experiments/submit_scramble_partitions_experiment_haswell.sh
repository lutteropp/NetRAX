#!/bin/sh

#source /etc/profile.d/modules.sh

#SBATCH -B 2:8:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1

module purge
module load Python
module load gompi/2019a
module load CMake/3.13.3-GCCcore-8.2.0
module load slurm

USAGE="Usage: sh submit_scramble_partitions_experiment_haswell.sh"

SCRIPTS="/home/luttersh/NetRAX/experiments/src"
FOLDER_PATH="/home/luttersh/NetRAX/experiments/data"
SUBFOLDER_PATH=${FOLDER_PATH}/${PREFIX}
LOGS_PATH=${FOLDER_PATH}/logs_${PREFIX}

python3 ${SCRIPTS}/scramble_partitions_experiment.py --prefix "D_scramble"

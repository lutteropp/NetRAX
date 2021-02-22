#!/bin/sh

module purge
module load CMake
module load Python
module load OpenMPI
module load slurm

#SBATCH -o something%j.log
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -B 2:8:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00

USAGE="Usage: sh run_experiments_bunch.sh SETTINGS PREFIX ITERATIONS"

if [ $# -lt 3 ]; then
    echo "Illegal number of parameters. ${USAGE}"
    exit 2
fi

SETTINGS=$1
PREFIX=$2
ITERATIONS=$3

FOLDER_PATH="data/"

mkdir ${PREFIX}_logs
i=0
while [ $i -lt ${ITERATIONS} ]; do
    python3 run_experiments.py --folder_path ${FOLDER_PATH} --labeled_settings ${SETTINGS} --prefix ${PREFIX}_${i} | tee ${PREFIX}_logs/${PREFIX}_${i}.log &
    i=$((i + 1))
done

wait

i=0
while [ $i -lt ${ITERATIONS} ]; do
	cat ${PREFIX}_logs/${PREFIX}_${i}.log
	i=$((i + 1))
done

python3 csv_merger.py --prefix ${PREFIX} --iterations ${ITERATIONS}

#!/bin/sh

#source /etc/profile.d/modules.sh

#SBATCH -B 2:8:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -N 10
#SBATCH -n 10

module purge
module load Python
module load gompi/2019a
module load CMake/3.13.3-GCCcore-8.2.0
module load slurm

USAGE="Usage: sh run_experiments_bunch.sh SETTINGS PREFIX ITERATIONS"

if [ $# -lt 3 ]; then
    echo "Illegal number of parameters. ${USAGE}"
    exit 2
fi

SETTINGS=$1
PREFIX=$2
ITERATIONS=$3

FOLDER_PATH="data/"
SUBFOLDER_PATH=${FOLDER_PATH}${PREFIX}
LOGS_PATH=${FOLDER_PATH}logs_${PREFIX}

[ ! -d ${FOLDER_PATH} ] && mkdir -p ${FOLDER_PATH}
[ ! -d ${LOGS_PATH} ] && mkdir -p ${LOGS_PATH}

rm -f ${LOGS_PATH}/${PREFIX}.log

i=0
while [ $i -lt ${ITERATIONS} ]; do
    python3 run_experiments.py --folder_path ${FOLDER_PATH} --labeled_settings ${SETTINGS} --prefix ${PREFIX}_${i} | tee ${LOGS_PATH}/${PREFIX}_${i}.log
    i=$((i + 1))
done

i=0
while [ $i -lt ${ITERATIONS} ]; do
	cat ${LOGS_PATH}/${PREFIX}_${i}.log >> ${LOGS_PATH}/${PREFIX}.log
	i=$((i + 1))
done

python3 csv_merger.py --prefix ${FOLDER_PATH}${PREFIX} --iterations ${ITERATIONS}

python3 postprocess_results.py --prefix ${FOLDER_PATH}${PREFIX} --iterations_global ${ITERATIONS}

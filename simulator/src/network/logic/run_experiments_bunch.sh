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

USAGE="Usage: sh run_experiments_bunch.sh PREFIX ITERATIONS MIN_TAXA MAX_TAXA MIN_RETICULATIONS MAX_RETICULATIONS MIN_RETICULATION_PROB MAX_RETICULATION_PROB [no_random]"

if [ $# -lt 8 ]; then
    echo "Illegal number of parameters. ${USAGE}"
    exit 2
fi

PREFIX=$1
ITERATIONS=$2
MIN_TAXA=$3
MAX_TAXA=$4
MIN_RETICULATIONS=$5
MAX_RETICULATIONS=$6
MIN_RETICULATION_PROB=$7
MAX_RETICULATION_PROB=$8

SAMPLING_TYPES="PERFECT_SAMPLING"
START_TYPES="FROM_RAXML RANDOM"
SIMULATOR_TYPES="CELINE"
BRLEN_LINKAGE_TYPES="LINKED"
LIKELIHOOD_TYPES="AVERAGE BEST"
PARTITION_SIZES="1000"
BRLEN_SCALERS="1.0 2.0"
FOLDER_PATH="data/"

if [ $# -eq 9 ]; then
    if [ $9=="no_random" ]; then
        START_TYPES="FROM_RAXML"
    else
        echo "Illegal third argument. Usage: ${USAGE}"
        exit 2
    fi
fi

mkdir ${PREFIX}_logs
i=0
while [ $i -lt ${ITERATIONS} ]; do
    python3 run_experiments.py --folder_path ${FOLDER_PATH} --prefix ${PREFIX}_${i} --sampling_types ${SAMPLING_TYPES} --start_types ${START_TYPES} --simulator_types ${SIMULATOR_TYPES} --brlen_scalers ${BRLEN_SCALERS} --brlen_linkage_types ${BRLEN_LINKAGE_TYPES} --likelihood_types ${LIKELIHOOD_TYPES} --partition_sizes ${PARTITION_SIZES} --min_taxa ${MIN_TAXA} --max_taxa ${MAX_TAXA} --min_reticulations ${MIN_RETICULATIONS} --max_reticulations ${MAX_RETICULATIONS} --min_reticulation_prob ${MIN_RETICULATION_PROB} --max_reticulation_prob ${MAX_RETICULATION_PROB} | tee ${PREFIX}_logs/${PREFIX}_${i}.log &
    i=$((i + 1))
done

wait

i=0
while [ $i -lt ${ITERATIONS} ]; do
	cat ${PREFIX}_logs/${PREFIX}_${i}.log
	i=$((i + 1))
done

python3 csv_merger.py --prefix ${PREFIX} --iterations ${ITERATIONS}

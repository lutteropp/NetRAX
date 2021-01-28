#!/bin/bash

#SBATCH -o experiments_%j.out
#SBATCH -N 1
#SBATCH -n 1 
#SBATCH -B 2:8:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00
 
source /etc/profile.d/modules.sh

USAGE="Usage: sh submit_experiments_larger_bunch_haswell.sh PREFIX BUNCHES ITERATIONS_PER_BUNCH MIN_TAXA MAX_TAXA MIN_RETICULATIONS MAX_RETICULATIONS [no_random]"

if [ $# -lt 7 ]; then
    echo "Illegal number of parameters. Usage: ${USAGE}"
    exit 2
fi

PREFIX=$1
BUNCHES=$2
ITERATIONS_PER_BUNCH=$3
MIN_TAXA=$4
MAX_TAXA=$5
MIN_RETICULATIONS=$6
MAX_RETICULATIONS=$7

SAMPLING_TYPES="PERFECT_SAMPLING"
START_TYPES="FROM_RAXML RANDOM"
BRLEN_LINKAGE_TYPES="LINKED"
LIKELIHOOD_TYPES="AVERAGE BEST"
PARTITION_SIZES="50 100"

i=0
while [ $i -lt ${BUNCHES} ]; do
    if [ $# -eq 8 ]; then
        if [ $8=="no_random" ]; then
            srun -N 1 -n 1 run_experiments_bunch.sh ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS} no_random &
        else
            echo "Unknown argument: ${8}"
            exit 2
        fi
    else
        srun -N 1 -n 1 run_experiments_bunch.sh ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS} &
    fi
    i=$((i + 1))
done

wait

python3 csv_merger.py --prefix ${PREFIX} --iterations ${BUNCHES}
python3 create_plots.py --prefix ${PREFIX}

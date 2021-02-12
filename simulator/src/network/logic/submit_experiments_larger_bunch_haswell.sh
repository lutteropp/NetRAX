#!/bin/bash
 
#source /etc/profile.d/modules.sh

#SBATCH -B 2:8:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00

module purge
module load CMake
module load Python
module load OpenMPI
module load slurm
module load Java

USAGE="Usage: sh submit_experiments_larger_bunch_haswell.sh PREFIX BUNCHES ITERATIONS_PER_BUNCH MIN_TAXA MAX_TAXA MIN_RETICULATIONS MAX_RETICULATIONS MIN_RETICULATION_PROB MAX_RETICULATION_PROB [no_random]"

if [ $# -lt 9 ]; then
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
MIN_RETICULATION_PROB=$8
MAX_RETICULATION_PROB=$9

SAMPLING_TYPES="PERFECT_SAMPLING"
START_TYPES="FROM_RAXML RANDOM"
BRLEN_LINKAGE_TYPES="LINKED"
LIKELIHOOD_TYPES="AVERAGE BEST"
SIMULATION_TYPES="CELINE"
PARTITION_SIZES="1000"
BRLEN_SCALERS="1.0 2.0"

i=0
while [ $i -lt ${BUNCHES} ]; do
    if [ $# -eq 10 ]; then
        if [ $10=="no_random" ]; then
            srun -N 1 -n 1 run_experiments_bunch.sh ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS} ${MIN_RETICULATION_PROB} ${MAX_RETICULATION_PROB} no_random &
        else
            echo "Unknown argument: ${10}"
            exit 2
        fi
    else
        srun -N 1 -n 1 run_experiments_bunch.sh ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS} ${MIN_RETICULATION_PROB} ${MAX_RETICULATION_PROB} &
    fi
    i=$((i + 1))
done

wait

#ln -f slurm-${SLURM_JOB_ID}.out ${PREFIX}.log
#rm slurm-${SLURM_JOB_ID}.out

python3 csv_merger.py --prefix ${PREFIX} --iterations ${BUNCHES}
python3 postprocess_results.py --prefix ${PREFIX}

#!/bin/bash
 
#source /etc/profile.d/modules.sh

#SBATCH -B 2:8:1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -N 10
#SBATCH -n 10

module purge
module load CMake
module load Python
module load OpenMPI
module load slurm
module load Java

USAGE="Usage: sh submit_experiments_larger_bunch_haswell.sh PREFIX BUNCHES ITERATIONS_PER_BUNCH"

if [ $# -lt 3 ]; then
    echo "Illegal number of parameters. Usage: ${USAGE}"
    exit 2
fi

PREFIX=$1
BUNCHES=10
ITERATIONS_PER_BUNCH=$3
FOLDER_PATH="data/"

i=0
while [ $i -lt ${BUNCHES} ]; do
    srun -N 1 -n 1 run_experiments_bunch.sh ${PREFIX} ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} &
    i=$((i + 1))
done

wait

#ln -f slurm-${SLURM_JOB_ID}.out ${PREFIX}.log
#rm slurm-${SLURM_JOB_ID}.out

python3 postprocess_results.py --prefix ${FOLDER_PATH}${PREFIX} --iterations_global ${BUNCHES} --iterations_local ${ITERATIONS_PER_BUNCH}

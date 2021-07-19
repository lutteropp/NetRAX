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

USAGE="Usage: sh submit_score_fixes_haswell.sh"

python3 src/repair_network_scores.py --prefix A_10_1
python3 src/repair_network_scores.py --prefix A_20_2
python3 src/repair_network_scores.py --prefix A_30_3
python3 src/repair_network_scores.py --prefix A_norandom_40_1
python3 src/repair_network_scores.py --prefix A_norandom_40_2
python3 src/repair_network_scores.py --prefix A_norandom_40_3
python3 src/repair_network_scores.py --prefix A_norandom_40_4
python3 src/repair_network_scores.py --prefix B_20_1
python3 src/repair_network_scores.py --prefix C_20_1

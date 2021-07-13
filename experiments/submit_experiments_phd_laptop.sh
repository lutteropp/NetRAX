#!/bin/sh

USAGE="Usage: sh submit_experiments_phd_laptop.sh SETTINGS ITERATIONS"

if [ $# -lt 2 ]; then
    echo "Illegal number of parameters. ${USAGE}"
    exit 2
fi

SETTINGS=$1
PREFIX=$1
ITERATIONS=$2

#SCRIPTS="/home/sarah/eclipse-workspace/NetRAX/experiments/src"
#FOLDER_PATH="/home/sarah/eclipse-workspace/NetRAX/experiments/data"

SCRIPTS="/home/luttersh/NetRAX/experiments/src"
FOLDER_PATH="/home/luttersh/NetRAX/experiments/data"
SUBFOLDER_PATH=${FOLDER_PATH}/${PREFIX}
LOGS_PATH=${FOLDER_PATH}/logs_${PREFIX}

[ ! -d ${FOLDER_PATH} ] && mkdir -p ${FOLDER_PATH}
[ ! -d ${LOGS_PATH} ] && mkdir -p ${LOGS_PATH}

rm -f ${LOGS_PATH}/${PREFIX}.log

i=0
while [ $i -lt ${ITERATIONS} ]; do
    python3 ${SCRIPTS}/run_experiments.py --folder_path ${FOLDER_PATH} --labeled_settings ${SETTINGS} --prefix ${PREFIX}_${i} | tee ${LOGS_PATH}/${PREFIX}_${i}.log
    i=$((i + 1))
done

i=0
while [ $i -lt ${ITERATIONS} ]; do
	cat ${LOGS_PATH}/${PREFIX}_${i}.log >> ${LOGS_PATH}/${PREFIX}.log
	i=$((i + 1))
done

python3 ${SCRIPTS}/csv_merger.py --prefix ${FOLDER_PATH}/${PREFIX} --iterations ${ITERATIONS}

python3 ${SCRIPTS}/postprocess_results.py --prefix ${FOLDER_PATH}/${PREFIX} --iterations_global ${ITERATIONS}

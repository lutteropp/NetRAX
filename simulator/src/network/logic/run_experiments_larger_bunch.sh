#!/bin/sh

USAGE="Usage: sh run_experiments_larger_bunch.sh PREFIX BUNCHES ITERATIONS_PER_BUNCH"

if [ $# -lt 3 ]; then
    echo "Illegal number of parameters. Usage: ${USAGE}"
    exit 2
fi

PREFIX=$1
BUNCHES=$2
ITERATIONS_PER_BUNCH=$3

FOLDER_PATH="data/"

i=0
while [ $i -lt ${BUNCHES} ]; do
    sh run_experiments_bunch.sh ${PREFIX} ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} &
    i=$((i + 1))
done

wait

python3 csv_merger.py --prefix ${PREFIX} --iterations ${BUNCHES}
python3 postprocess_results.py --prefix ${PREFIX} --iterations_global ${BUNCHES} --iterations_local ${ITERATIONS_PER_BUNCH}

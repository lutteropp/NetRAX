#!/bin/sh

ITERATIONS_PER_BUNCH=2

USAGE="Usage: sh run_experiments_larger_bunch.sh PREFIX BUNCHES MIN_TAXA MAX_TAXA MIN_RETICULATIONS MAX_RETICULATIONS [no_random]"

if [ $# -lt 6 ]; then
    echo "Illegal number of parameters. Usage: ${USAGE}"
    exit 2
fi

PREFIX=$1
BUNCHES=$2
MIN_TAXA=$3
MAX_TAXA=$4
MIN_RETICULATIONS=$5
MAX_RETICULATIONS=$6

SAMPLING_TYPES="PERFECT_SAMPLING"
START_TYPES="FROM_RAXML RANDOM"
BRLEN_LINKAGE_TYPES="LINKED"
LIKELIHOOD_TYPES="AVERAGE BEST"
PARTITION_SIZES="50 100"

if [ $# -eq 7 ]; then
    if [ $3=="no_random" ]; then
        START_TYPES="FROM_RAXML"
    else
        echo "Illegal third argument. Usage: ${USAGE}"
        exit 2
    fi
fi

i=0
while [ $i -lt ${BUNCHES} ]; do
    if [ $# -eq 7 -a $3=="no_random" ]; then
        sh run_experiments_bunch.sh ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS} no_random
    else
        sh run_experiments_bunch.sh ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS}
    fi
    i=$((i + 1))
done

wait

python3 csv_merger.py --prefix ${PREFIX} --iterations ${BUNCHES}
python3 create_plots.py --prefix ${PREFIX}

#!/bin/sh

USAGE="Usage: sh run_experiments_larger_bunch.sh PREFIX BUNCHES ITERATIONS_PER_BUNCH MIN_TAXA MAX_TAXA MIN_RETICULATIONS MAX_RETICULATIONS [no_random]"

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
            sh run_experiments_bunch.sh ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS} no_random &
        else
            echo "Unknown argument: ${8}"
            exit 2
        fi
    else
        sh run_experiments_bunch.sh ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS} &
    fi
    i=$((i + 1))
done

wait

python3 csv_merger.py --prefix ${PREFIX} --iterations ${BUNCHES}
python3 postprocess_results.py --prefix ${PREFIX}

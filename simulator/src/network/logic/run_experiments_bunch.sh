#!/bin/sh

USAGE="Usage: sh run_experiments_bunch.sh PREFIX ITERATIONS MIN_TAXA MAX_TAXA MIN_RETICULATIONS MAX_RETICULATIONS [no_random]"

if [ $# -lt 6 ]; then
    echo "Illegal number of parameters. ${USAGE}"
    exit 2
fi

PREFIX=$1
ITERATIONS=$2
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
    if [ $7=="no_random" ]; then
        START_TYPES="FROM_RAXML"
    else
        echo "Illegal third argument. Usage: ${USAGE}"
        exit 2
    fi
fi

i=0
while [ $i -lt ${ITERATIONS} ]; do
    python3 run_experiments.py --prefix ${PREFIX}_${i} --sampling_types ${SAMPLING_TYPES} --start_types ${START_TYPES} --brlen_linkage_types ${BRLEN_LINKAGE_TYPES} --likelihood_types ${LIKELIHOOD_TYPES} --partition_sizes ${PARTITION_SIZES} --min_taxa ${MIN_TAXA} --max_taxa ${MAX_TAXA} --min_reticulations ${MIN_RETICULATIONS} --max_reticulations ${MAX_RETICULATIONS} &
    i=$((i + 1))
done

wait

python3 csv_merger.py --prefix ${PREFIX} --iterations ${ITERATIONS}
#python3 postprocess_results.py --prefix ${PREFIX}

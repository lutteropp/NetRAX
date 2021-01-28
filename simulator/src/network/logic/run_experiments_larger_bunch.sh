#!/bin/sh

ITERATIONS_PER_BUNCH=2

USAGE="Usage: sh run_experiment_bunch.sh PREFIX BUNCHES MIN_TAXA MAX_TAXA MIN_RETICULATIONS MAX_RETICULATIONS [no_random]"

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

case ${PREFIX} in
"small_network")
    MIN_TAXA=4
    MAX_TAXA=10
    MIN_RETICULATIONS=1
    MAX_RETICULATIONS=2
    ;;
"small_network_debug")
    MIN_TAXA=4
    MAX_TAXA=4
    MIN_RETICULATIONS=1
    MAX_RETICULATIONS=2
    ;;
"larger_network")
    MIN_TAXA=20
    MAX_TAXA=50
    MIN_RETICULATIONS=2
    MAX_RETICULATIONS=-1
    ;;
"small_tree")
    MIN_TAXA=4
    MAX_TAXA=10
    MIN_RETICULATIONS=0
    MAX_RETICULATIONS=0
    ;;
*)
    echo "Unknown prefix"
    exit 2
    ;;
esac

i=0
while [ $i -lt ${BUNCHES} ]; do
    if [ $# -eq 7 && $3=="no_random" ]; then
        sh run_experiment_bunch ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS} no_random
    else
        sh run_experiment_bunch ${PREFIX}_${i} ${ITERATIONS_PER_BUNCH} ${MIN_TAXA} ${MAX_TAXA} ${MIN_RETICULATIONS} ${MAX_RETICULATIONS}
    fi
    i=$((i + 1))
done

wait

python3 csv_merger.py --prefix ${PREFIX} --iterations ${BUNCHES}
python3 create_plots.py --prefix ${PREFIX}

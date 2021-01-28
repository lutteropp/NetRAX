#!/bin/sh

if [ $# -lt 2 ]; then
    echo "Illegal number of parameters. Usage: sh run_experiment_bunch.sh PREFIX ITERATIONS [no_random]"
    exit 2
fi

PREFIX=$1
ITERATIONS=$2
SAMPLING_TYPES="SamplingType.PERFECT_SAMPLING"
START_TYPES="StartType.FROM_RAXML StartType.RANDOM"
BRLEN_LINKAGE_TYPES="BrlenLinkageType.LINKED"
LIKELIHOOD_TYPES="LikelihoodType.AVERAGE LikelihoodType.BEST"
PARTITION_SIZES="50 100"

if [ $# -eq 3 ]; then
    if [ $3=="no_random" ]; then
        START_TYPES="StartType.FROM_RAXML"
    else
        echo "Illegal third argument. Usage: sh run_experiment_bunch.sh PREFIX ITERATIONS [no_random]"
        exit 2
    fi
fi

case ${PREFIX} in
"small_network") MIN_TAXA=4; MAX_TAXA=10; MIN_RETICULATIONS=1; MAX_RETICULATIONS=2;;
"small_network_debug") MIN_TAXA=4; MAX_TAXA=4; MIN_RETICULATIONS=1; MAX_RETICULATIONS=2;;
"larger_network") MIN_TAXA=20; MAX_TAXA=50; MIN_RETICULATIONS=2; MAX_RETICULATIONS=-1;;
"small_tree") MIN_TAXA=4; MAX_TAXA=10; MIN_RETICULATIONS=0; MAX_RETICULATIONS=0;;
*) echo "Unknown prefix"; exit 2;;
esac

i=0
while [ $i -lt ${ITERATIONS} ]; do
    python3 run_experiments.py --prefix ${PREFIX}_${i} --iterations 1 --sampling_types ${SAMPLING_TYPES} --start_types ${START_TYPES} --brlen_linkage_types ${BRLEN_LINKAGE_TYPES} --likelihood_types ${LIKELIHOOD_TYPES} --partition_sizes ${PARTITION_SIZES} --min_taxa ${MIN_TAXA} --max_taxa ${MAX_TAXA} --min_reticulations ${MIN_RETICULATIONS} --max_reticulations ${MAX_RETICULATIONS} &
    i=$(( i + 1 ))
done

wait

python3 csv_merger.py --prefix ${PREFIX} --iterations ${ITERATIONS}
python3 create_plots.py --prefix ${PREFIX}

#!/usr/bin/env bash

ITERATIONS = $2
PREFIX = $1

SAMPLING_TYPES = ""
START_TYPES = ""
BRLEN_LINKAGE_TYPES = ""
LIKELIHOOD_TYPES = ""
PARTITION_SIZES = ""
MIN_TAXA = ""
MAX_TAXA = ""
MIN_RETICULATIONS = ""
MAX_RETICULATIONS = ""

for ((i = 0; i < ${ITERATIONS}; i++)); do
    python3 run_experiments.py --prefix ${PREFIX}_${i} --iterations 1 --sampling_types ${SAMPLING_TYPES} --start_types ${START_TYPES} --brlen_linkage_types ${BRLEN_LINKAGE_TYPES} --likelihood_types ${LIKELIHOOD_TYPES} --partition_sizes ${PARTITION_SIZES} --min_taxa ${MIN_TAXA} --max_taxa ${MAX_TAXA} --min_reticulations ${MIN_RETICULATIONS} --max_reticulations ${MAX_RETICULATIONS} &
done

wait

python3 csv_merger.py --prefix ${PREFIX} --iterations ${ITERATIONS}
python3 create_plots.py --prefix ${PREFIX}

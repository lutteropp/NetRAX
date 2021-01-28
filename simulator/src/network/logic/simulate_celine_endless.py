from celine_simulator import CelineParams, simulate_network

import jsonpickle
from collections import defaultdict
import time

ITERATIONS = 100


class CelineResult:
    def __init__(self):
        self.n_taxa = 0
        self.n_reticulations = 0
        self.newick = ""
        self.param_info = {}


counter = dict()
for taxa in range(1001):
    counter[taxa] = defaultdict(int)

t0 = time.time()
celine_results = []
for _ in range(ITERATIONS):
    params = CelineParams()
    n_taxa, n_reticulations, newick, param_info = simulate_network(params)
    res = CelineResult()
    res.n_taxa = n_taxa
    res.n_reticulations = n_reticulations
    res.newick = newick
    res.param_info = param_info

    celine_results.append(res)
    counter[n_taxa][n_reticulations] += 1
t1 = time.time() - t0
# CPU seconds elapsed (floating point)
print("Time elapsed for running simulations: ", t1)
print("Average simulation time: " + str(float(t1/ITERATIONS)))

for i in range(1001):
    for j in range(1001):
        if counter[i][j] != 0:
            print(str(i) + ", " + str(j) + ": " + str(counter[i][j]))

with open('celine_simulated_datasets.json', 'w') as f:
    f.write(jsonpickle.encode(celine_results))

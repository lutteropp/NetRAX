import subprocess
import os

NETRAX_PATH = "/home/sarah/code-workspace/NetRAX/bin/netrax"


# Uses netrax to compute the number of reticulations, BIC score, and loglikelihood of a network for a given MSA
def score_network(network_path, msa_path):
    netrax_cmd = NETRAX_PATH + " --score_only" + " --start_network" + network_path + " --msa " + msa_path
    netrax_output = subprocess.getoutput(netrax_cmd).splitlines()
    n_reticulations, bic, logl = 0,0,0
    for line in netrax_output:
       if line.startswith("Number of reticulations:"):
            n_reticulations = float(line.split(": ")[1])
       if line.startswith("BIC Score:"):
            bic = float(line.split(": ")[1])
       if line.startswith("Loglikelihood:"):
            logl = float(line.split(": ")[1])
    return n_reticulations, bic, logl

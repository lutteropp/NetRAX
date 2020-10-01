import subprocess
import os

NETRAX_PATH = "/home/sarah/code-workspace/NetRAX/bin/netrax"


# Uses NetRAX to compute the number of reticulations, BIC score, and loglikelihood of a network for a given MSA
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
    
    
# Uses NetRAX to infer a network... uses single random starting network if timeout==0, else keeps searching for a better network until timeout seconds have passed.
def infer_network(msa_path, output_path, timeout):
    netrax_cmd = NETRAX_PATH + " --msa " + msa_path + " --output " + output_path
    if timeout > 0:
        netrax_cmd += " --endless --timeout " + str(timeout)
    subprocess.getoutput(netrax_cmd).splitlines()

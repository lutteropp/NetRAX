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
    
    
# Extracts all displayed trees of a given network, returning two lists: one containing the NEWICK strings, and one containing the tree probabilities
def extract_displayed_trees(network_path):
    netrax_cmd = NETRAX_PATH + " --extract_displayed_trees " + " --start_network " + network_path
    lines = subprocess.getoutput(netrax_cmd).splitlines()
    start_idx = 0
    n_trees = 0
    for i in range(len(lines)):
        if lines[i].startswith("Number of displayed trees:"):
            n_trees = int(lines[i].split(": ")[1])
            start_idx = i + 1
            break
    trees_newick = []
    trees_prob = []
    for i in range(n_trees):
        trees_newick.append(lines[start_idx + i].replace('\n',''))
    start_idx += n_trees + 1
    for i in range(n_trees):
        trees_prob.append(float(lines[start_idx + i].split(": ")[1]))
    return trees_newick, trees_prob
    
    
# Generates a random network with the wanted number of taxa and reticulations. Writes it in Extended NEWICK format to the provided output path.
def generate_random_network(n_taxa, n_reticulations, output_path):
    msa_path = "temp_fake_msa.txt"
    msa_file = open(msa_path, "w")
    for i in range(n_taxa):
        msa_file.write(">T" + str(i) + "\n" + "A\n")
    msa_file.close()
    netrax_cmd = NETRAX_PATH + " --generate_random_network_only " + " --max_reticulations " + str(n_reticulations) + " --msa " + msa_path + " --output " + output_path
    subprocess.getoutput(netrax_cmd)
    os.remove(msa_path)

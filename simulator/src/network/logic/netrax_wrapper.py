import subprocess
import os
import time

from experiment_model import LikelihoodType, InferenceType

NETRAX_PATH = "/home/sarah/code-workspace/NetRAX/bin/netrax"


# Uses NetRAX to compute the number of reticulations, BIC score, and loglikelihood of a network for a given MSA
def score_network(network_path, msa_path, partitions_path, likelihood_type):
    netrax_cmd = NETRAX_PATH + " --score_only" + " --start_network " + network_path + " --msa " + msa_path + " --model " + partitions_path
    if likelihood_type == LikelihoodType.BEST:
        netrax_cmd += " --best_displayed_tree_variant"
    print(netrax_cmd)
    netrax_output = subprocess.getoutput(netrax_cmd).splitlines()
    print(netrax_output)
    n_reticulations, bic, logl = 0,0,0
    for line in netrax_output:
       if line.startswith("Number of reticulations:"):
            n_reticulations = float(line.split(": ")[1])
       if line.startswith("BIC Score:"):
            bic = float(line.split(": ")[1])
       if line.startswith("Loglikelihood:"):
            logl = float(line.split(": ")[1])
    return n_reticulations, bic, logl
    
    
# Uses NetRAX to infer a network... uses few random starting network if timeout==0, else keeps searching for a better network until timeout seconds have passed.
def infer_network(ds):
    runtime_inference = 0
    runtime_inference_with_raxml = 0
    
    if ds.inference_type != InferenceType.FROM_RAXML_ONLY:
        netrax_cmd = NETRAX_PATH + " --msa " + ds.msa_path + " --model " + ds.partitions_path + " --output " + ds.inferred_network_path
        if ds.likelihood_type == LikelihoodType.BEST:
            netrax_cmd += " --best_displayed_tree_variant"
        if ds.timeout > 0:
            netrax_cmd += " --endless --timeout " + str(ds.timeout)
        else:
            netrax_cmd += " --num_random_start_networks " + str(ds.n_random_start_networks) + " --num_parsimony_start_networks " + str(ds.n_parsimony_start_networks)
        print(netrax_cmd)
        
        start_normal = time.time()
        print(subprocess.getoutput(netrax_cmd))
        runtime_inference = round(time.time() - start_normal, 3)
    
    if ds.start_from_raxml:
        netrax_cmd_2 = NETRAX_PATH + " --msa " + ds.msa_path + " --model " + ds.partitions_path + " --output " + ds.inferred_network_with_raxml_path + " --start_network " + ds.raxml_tree_path
        if ds.likelihood_type == LikelihoodType.BEST:
            netrax_cmd_2 += " --best_displayed_tree_variant"
        print(netrax_cmd_2)
 
        start_with_raxml = time.time()
        print(subprocess.getoutput(netrax_cmd_2))
        runtime_inference_with_raxml = round(time.time() - start_with_raxml, 3)
    return runtime_inference, runtime_inference_with_raxml
    
    
# Extracts all displayed trees of a given network, returning two lists: one containing the NEWICK strings, and one containing the tree probabilities
def extract_displayed_trees(network_path, n_taxa):
    msa_path = "temp_fake_msa.txt"
    msa_file = open(msa_path, "w")
    msa_file.write(build_fake_msa(n_taxa, network_path))
    msa_file.close()

    netrax_cmd = NETRAX_PATH + " --extract_displayed_trees " + " --start_network " + network_path + " --msa " + msa_path + " --model DNA"
    print(netrax_cmd)
    lines = subprocess.getoutput(netrax_cmd).splitlines()
    print(lines)
    start_idx = 0
    n_trees = 0
    for i in range(len(lines)):
        if lines[i].startswith("Number of displayed trees:"):
            n_trees = int(lines[i].split(": ")[1])
            start_idx = i + 2
            break
    trees_newick = []
    trees_prob = []
    for i in range(n_trees):
        trees_newick.append(lines[start_idx + i].replace('\n',''))
    start_idx += n_trees + 1
    for i in range(n_trees):
        trees_prob.append(float(lines[start_idx + i]))
    #os.remove(msa_path)
    return trees_newick, trees_prob


def build_fake_msa(n_taxa, network_path=""):
    fake_msa = ""
    
    def to_dna(s):
        BS = "ACGT"
        res = ""
        while s:
            res+=BS[s%4]
            s//= 4
        return res[::-1] or "A"
    
    msa = [to_dna(i) for i in range(n_taxa)]
    maxlen = max([len(s) for s in msa])
    msa = [s.rjust(maxlen, 'A') for s in msa]
    
    taxon_names = []
    if network_path != "":
        netrax_cmd = NETRAX_PATH + " --extract_taxon_names " + " --start_network " + network_path  + " --model DNA"
        print(netrax_cmd)
        lines = subprocess.getoutput(netrax_cmd).splitlines()
        for line in lines[1:]:
            taxon_names.append(line)
            
    for i in range(n_taxa):
        if len(taxon_names) == 0:
            fake_msa += ">T" + str(i) + "\n" + msa[i] + "\n"
        else:
            fake_msa += ">" + taxon_names[i] + "\n" + msa[i] + "\n"
    print(fake_msa)
    return fake_msa    

    
# Generates a random network with the wanted number of taxa and reticulations. Writes it in Extended NEWICK format to the provided output path.
def generate_random_network(n_taxa, n_reticulations, output_path):
    msa_path = "temp_fake_msa.txt"
    msa_file = open(msa_path, "w")
    msa_file.write(build_fake_msa(n_taxa))
    msa_file.close()
    netrax_cmd = NETRAX_PATH + " --generate_random_network_only " + " --max_reticulations " + str(n_reticulations) + " --msa " + msa_path + " --model DNA " + " --output " + output_path
    print(netrax_cmd)
    print(subprocess.getoutput(netrax_cmd))
    os.remove(msa_path)

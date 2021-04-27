import subprocess
import os
import time
import random

from experiment_model import LikelihoodType, BrlenLinkageType, StartType

NETRAX_PATH = "../../../../bin/netrax"

NETWORK_DISTANCE_NAMES = ['unrooted_softwired_network_distance', 'unrooted_hardwired_network_distance', 'unrooted_displayed_trees_distance', 'rooted_softwired_network_distance', 'rooted_hardwired_network_distance', 'rooted_displayed_trees_distance', 'rooted_tripartition_distance', 'rooted_path_multiplicity_distance', 'rooted_nested_labels_distance']

# Uses NetRAX to compute the number of reticulations, BIC score, and loglikelihood of a network for a given MSA
def score_network(network_path, msa_path, partitions_path, likelihood_type, brlen_linkage_type):
    netrax_cmd = NETRAX_PATH + " --score_only" + " --start_network " + \
        network_path + " --msa " + msa_path + " --model " + partitions_path

    if likelihood_type == LikelihoodType.AVERAGE:
        netrax_cmd += " --average_displayed_tree_variant"
    else:
        netrax_cmd += " --best_displayed_tree_variant"

    if brlen_linkage_type == BrlenLinkageType.UNLINKED:
        netrax_cmd += " --brlen unlinked"
    elif brlen_linkage_type == BrlenLinkageType.LINKED:
        netrax_cmd += " --brlen linked"
    else:
        netrax_cmd += " --brlen scaled"
    netrax_cmd += " --seed 42"

    print(netrax_cmd)
    cmd_status, cmd_output = subprocess.getstatusoutput(netrax_cmd)
    print(cmd_output)
    if cmd_status != 0:
        raise Exception("Scoring network failed")
    netrax_output = cmd_output.splitlines()
    n_reticulations, bic, logl = 0, 0, 0
    for line in netrax_output:
        if line.startswith("Number of reticulations:"):
            n_reticulations = float(line.split(": ")[1])
        if line.startswith("BIC Score:"):
            bic = float(line.split(": ")[1])
        if line.startswith("Loglikelihood:"):
            logl = float(line.split(": ")[1])
    return n_reticulations, bic, logl


def infer_networks(ds):
    netrax_cmd_start = NETRAX_PATH + " --msa " + ds.msa_path
    for var in ds.inference_variants:
        netrax_cmd = netrax_cmd_start + " --output " + var.inferred_network_path

        if var.use_partitioned_msa:
            netrax_cmd += " --model " + ds.partitions_path
        else:
            netrax_cmd += " --model DNA"

        if var.likelihood_type == LikelihoodType.AVERAGE:
            netrax_cmd += " --average_displayed_tree_variant"
        else:
            netrax_cmd += " --best_displayed_tree_variant"

        if var.start_type == StartType.ENDLESS:
            netrax_cmd += " --endless --timeout " + str(var.timeout)
        elif var.start_type == StartType.RANDOM:
            netrax_cmd += " --num_random_start_networks " + \
                str(var.n_random_start_networks) + " --num_parsimony_start_networks " + \
                str(var.n_parsimony_start_networks)
        else:  # StartType.FROM_RAXML
            netrax_cmd += " --start_network " + ds.raxml_tree_path

        if var.brlen_linkage_type == BrlenLinkageType.UNLINKED:
            netrax_cmd += " --brlen unlinked"
        elif var.brlen_linkage_type == BrlenLinkageType.LINKED:
            netrax_cmd += " --brlen linked"
        else:
            netrax_cmd += " --brlen scaled"
        netrax_cmd += " --seed 42"

        print(netrax_cmd, flush=True)
        start_time = time.time()

        cmd_status, cmd_output = subprocess.getstatusoutput(netrax_cmd)
        print(cmd_output)
        if cmd_status != 0:
            raise Exception("Inferring network failed")

        var.runtime_inference = round(time.time() - start_time, 3)


# Extracts all displayed trees of a given network, returning two lists: one containing the NEWICK strings, and one containing the tree probabilities
def extract_displayed_trees(network_path, n_taxa):
    msa_path = "temp_fake_msa_" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".txt"
    msa_file = open(msa_path, "w")
    msa_file.write(build_fake_msa(n_taxa, network_path))
    msa_file.close()

    netrax_cmd = NETRAX_PATH + " --extract_displayed_trees" + \
        " --start_network " + network_path + " --msa " + msa_path + " --model DNA"
    print(netrax_cmd, flush=True)
    cmd_status, cmd_output = subprocess.getstatusoutput(netrax_cmd)
    print(cmd_output)
    if cmd_status != 0:
        raise Exception("Extract displayed trees failed")
    lines = cmd_output.splitlines()
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
        trees_newick.append(lines[start_idx + i].replace('\n', ''))
    start_idx += n_trees + 1
    for i in range(n_trees):
        trees_prob.append(float(lines[start_idx + i]))
    os.remove(msa_path)
    return trees_newick, trees_prob
    

def change_reticulation_prob_only(network_input_path, network_output_path, new_ret_prob, n_taxa):
    msa_path = "temp_fake_msa_" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".txt"
    msa_file = open(msa_path, "w")
    msa_file.write(build_fake_msa(n_taxa, network_input_path))
    msa_file.close()
    netrax_cmd = NETRAX_PATH + " --change_reticulation_prob_only --overwritten_reticulation_prob " + str(new_ret_prob) + \
        " --start_network " + network_input_path + " --output " + network_output_path + " --msa " + msa_path + " --model DNA"
    print(netrax_cmd, flush=True)
    cmd_status, cmd_output = subprocess.getstatusoutput(netrax_cmd)
    print(cmd_output)
    if cmd_status != 0:
        raise Exception("Change reticulation prob failed")
    os.remove(msa_path)


def change_reticulation_prob_only_newick(newick, network_output_path, new_ret_prob, n_taxa):
    temp_network_input = "temp_network_input" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".nw"
    temp_network_input_file = open(temp_network_input, "w")
    temp_network_input_file.write(newick + "\n")
    temp_network_input_file.close()
    change_reticulation_prob_only(temp_network_input, network_output_path, new_ret_prob, n_taxa)
    os.remove(temp_network_input)


def network_distance_only(network_1_path, network_2_path, n_taxa):
    msa_path = "temp_fake_msa_" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".txt"
    msa_file = open(msa_path, "w")
    msa_file.write(build_fake_msa(n_taxa, network_1_path))
    msa_file.close()

    distances = {}
    for name in NETWORK_DISTANCE_NAMES:
        distances[name] = -1

    netrax_cmd = NETRAX_PATH + " --network_distance_only " + \
        " --first_network " + network_1_path + " --second_network " + network_2_path + " --msa " + msa_path + " --model DNA"
    print(netrax_cmd, flush=True)
    cmd_status, cmd_output = subprocess.getstatusoutput(netrax_cmd)
    print(cmd_output)
    lines = cmd_output.splitlines()
    dist = -1
    for line in lines:
        if line.startswith("Unrooted softwired network distance: "):
            distances['unrooted_softwired_network_distance'] = float(line.split(": ")[1])
        elif line.startswith("Unrooted hardwired network distance: "):
            distances['unrooted_hardwired_network_distance'] = float(line.split(": ")[1])
        elif line.startswith("Unrooted displayed trees distance: "):
            distances['unrooted_displayed_trees_distance'] = float(line.split(": ")[1])

        elif line.startswith("Rooted softwired network distance: "):
            distances['rooted_softwired_network_distance'] = float(line.split(": ")[1])
        elif line.startswith("Rooted hardwired network distance: "):
            distances['rooted_hardwired_network_distance'] = float(line.split(": ")[1])
        elif line.startswith("Rooted displayed trees distance: "):
            distances['rooted_displayed_trees_distance'] = float(line.split(": ")[1])
        elif line.startswith("Rooted tripartition distance: "):
            distances['rooted_tripartition_distance'] = float(line.split(": ")[1])
        elif line.startswith("Rooted path multiplicity distance: "):
            distances['rooted_path_multiplicity_distance'] = float(line.split(": ")[1])
        elif line.startswith("Rooted nested labels distance: "):
            distances['rooted_nested_labels_distance'] = float(line.split(": ")[1])
    os.remove(msa_path)
    return distances


def check_weird_network(network_path, n_taxa):
    msa_path = "temp_fake_msa_" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".txt"
    msa_file = open(msa_path, "w")
    msa_file.write(build_fake_msa(n_taxa, network_path))
    msa_file.close()

    netrax_cmd = NETRAX_PATH + " --check_weird_network " + \
        " --start_network " + network_path + " --msa " + msa_path + " --model DNA"
    print(netrax_cmd, flush=True)
    cmd_status, cmd_output = subprocess.getstatusoutput(netrax_cmd)
    print(cmd_output)
    if cmd_status != 0:
        raise Exception("Check weird network failed")
    lines = cmd_output.splitlines()

    n_pairs = 0
    n_equal = 0

    for i in range(len(lines)):
        if lines[i].startswith("Number of pairs:"):
            n_pairs = int(lines[i].split(": ")[1])
        elif lines[i].startswith("Number of equal pairs:"):
            n_equal = int(lines[i].split(": ")[1])

    os.remove(msa_path)
    return n_pairs, n_equal


def scale_branches_only(network_path, output_path, scaling_factor, n_taxa):
    msa_path = "temp_fake_msa_" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".txt"
    msa_file = open(msa_path, "w")
    msa_file.write(build_fake_msa(n_taxa, network_path))
    msa_file.close()

    netrax_cmd = NETRAX_PATH + " --scale_branches_only " + str(scaling_factor) + \
        " --start_network " + network_path + " --msa " + msa_path + " --model DNA" + " --output " + output_path
    print(netrax_cmd, flush=True)
    cmd_status, cmd_output = subprocess.getstatusoutput(netrax_cmd)
    print(cmd_output)
    if cmd_status != 0:
        raise Exception("Check weird network failed")

    os.remove(msa_path)


def scale_branches_only_newick(newick, output_path, scaling_factor, n_taxa):
    temp_network_input = "temp_network_input" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".nw"
    temp_network_input_file = open(temp_network_input, "w")
    temp_network_input_file.write(newick + "\n")
    temp_network_input_file.close()
    scale_branches_only(temp_network_input, output_path, scaling_factor, n_taxa)
    os.remove(temp_network_input)


def build_fake_msa(n_taxa, network_path=""):
    fake_msa = ""

    def to_dna(s):
        BS = "ACGT"
        res = ""
        while s:
            res += BS[s % 4]
            s //= 4
        return res[::-1] or "A"

    msa = [to_dna(i) for i in range(n_taxa)]
    maxlen = max([len(s) for s in msa])
    msa = [s.rjust(maxlen, 'A') for s in msa]

    taxon_names = []
    if network_path != "":
        netrax_cmd = NETRAX_PATH + " --extract_taxon_names " + \
            " --start_network " + network_path + " --model DNA"
        print(netrax_cmd, flush=True)
        cmd_status, cmd_output = subprocess.getstatusoutput(netrax_cmd)
        print(cmd_output)
        if cmd_status != 0:
            raise Exception("Extract taxon names failed")
        lines = cmd_output.splitlines()
        for line in lines[1:]:
            taxon_names.append(line)

    for i in range(n_taxa):
        if len(taxon_names) == 0:
            fake_msa += ">T" + str(i) + "\n" + msa[i] + "\n"
        else:
            fake_msa += ">" + taxon_names[i] + "\n" + msa[i] + "\n"
    #print(fake_msa)
    return fake_msa


# Generates a random network with the wanted number of taxa and reticulations. Writes it in Extended NEWICK format to the provided output path.
def generate_random_network(n_taxa, n_reticulations, output_path):
    msa_path = "temp_fake_msa_" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".txt"
    msa_file = open(msa_path, "w")
    msa_file.write(build_fake_msa(n_taxa))
    msa_file.close()
    netrax_cmd = NETRAX_PATH + " --generate_random_network_only " + " --max_reticulations " + \
        str(n_reticulations) + " --msa " + msa_path + \
        " --model DNA " + " --output " + output_path
    print(netrax_cmd, flush=True)
    cmd_status, cmd_output = subprocess.getstatusoutput(netrax_cmd)
    print(cmd_output)
    if cmd_status != 0:
        raise Exception("Generate random network failed")
    os.remove(msa_path)

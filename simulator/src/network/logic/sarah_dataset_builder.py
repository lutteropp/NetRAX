
from experiment_model import Dataset, SamplingType, SimulationType
from netrax_wrapper import generate_random_network, extract_displayed_trees
from seqgen_wrapper import simulate_msa

import math
import random

def create_dataset_container_sarah(n_taxa, n_reticulations, approx_msa_size, sampling_type, name, timeout=0, m=1):
    n_trees = 2 ** n_reticulations
    if sampling_type == SamplingType.STANDARD:
        n_trees *= m
    sites_per_tree = math.ceil(approx_msa_size / n_trees)
    msa_size = approx_msa_size
    
    if sampling_type == SamplingType.STANDARD:
        msa_size = sites_per_tree * n_trees
    if sampling_type == SamplingType.SINGLE_SITE_SAMPLING:
        sites_per_tree = 1
        n_trees = approx_msa_size
    ds = Dataset()
    ds.n_taxa = n_taxa
    ds.n_reticulations = n_reticulations
    ds.n_trees = n_trees
    ds.sites_per_tree = sites_per_tree
    ds.msa_path = name + "_msa.txt"
    ds.extracted_trees_path = name + "_trees.txt"
    ds.true_network_path = name + "_true_network.nw"
    ds.inferred_network_path = name + "_inferred_network.nw"
    ds.sampling_type = sampling_type
    ds.simulation_type = SimulationType.SARAH
    ds.timeout = timeout
    return ds
    
    
def sample_trees_sarah(ds, trees_prob):
    n_displayed_trees = len(trees_prob)
    sampled_trees_contrib = [0] * n_displayed_trees
    
    if ds.sampling_type == SamplingType.PERFECT_UNIFORM_SAMPLING:
        spt = math.ceil(ds.msa_size / n_displayed_trees)
        ds.msa_size = spt * n_displayed_trees
        sampled_trees_contrib = [spt] * n_displayed_trees
    elif ds.sampling_type == SamplingType.PERFECT_SAMPLING:
        sampled_trees_contrib = [math.ceil(p * ds.msa_size) for p in trees_prob]
        ds.msa_size = sum(sampled_trees_contrib)
    else: # standard sampling and single site sampling are actually the same, just the number of sampled sites per tree differs
        tree_indices = [i for i in range(n_displayed_trees)]
        for _ in range(ds.n_trees):
            tree_idx = random.choices(population=tree_indices, weights=trees_prob, k=1)[0]
            sampled_trees_contrib[tree_idx] += ds.sites_per_tree
    
    return ds, sampled_trees_contrib
    
    
# build the trees file required by seq-gen
def build_trees_file_sarah(ds, trees_newick, sampled_trees_contrib):
    trees_file = open(ds.extracted_trees_path, "w")
    for i in range(len(trees_newick)):
        newick = trees_newick[i]
        if sampled_trees_contrib[i] > 0:
            trees_file.write('[' + str(sampled_trees_contrib[i]) + ']' + newick + "\n")
    trees_file.close()
    
    
def build_dataset_sarah(n_taxa, n_reticulations, approx_msa_size, sampling_type, name, timeout=0, m=1):
    ds = create_dataset_container_sarah(n_taxa, n_reticulations, approx_msa_size, sampling_type, name, timeout, m)
    generate_random_network(ds.n_taxa, ds.n_reticulations, ds.true_network_path)
    trees_newick, trees_prob = extract_displayed_trees(ds.true_network_path, ds.n_taxa)
    ds, sampled_tree_contrib = sample_trees_sarah(ds, trees_prob)
    build_trees_file_sarah(ds, trees_newick, sampled_trees_contrib)
    simulate_msa(ds)
    return ds
   

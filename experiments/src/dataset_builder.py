
from experiment_model import Dataset, SamplingType, SimulatorType, LikelihoodType, BrlenLinkageType, StartType, InferenceVariant
from netrax_wrapper import generate_random_network, extract_displayed_trees, check_weird_network
from seqgen_wrapper import simulate_msa
from celine_simulator import CelineParams, simulate_network_celine

import math
import random


def create_dataset_container(n_taxa, n_reticulations, approx_msa_size, sampling_type, simulation_type, brlen_scaler, likelihood_types, brlen_linkage_types, start_types, part_msa_types, my_id, name, fixed_reticulation_prob=None, timeout=0, m=1, num_start_networks=3):
    ds = Dataset()
    ds.my_id = my_id
    ds.n_taxa = n_taxa
    ds.n_reticulations = n_reticulations
    ds.msa_path = name + "_msa.txt"
    ds.partitions_path = name + "_partitions.txt"
    ds.extracted_trees_path = name + "_trees.txt"
    ds.true_network_path = name + "_true_network.nw"
    ds.raxml_tree_path = name + ".raxml.bestTree"
    ds.name = name
    ds.sampling_type = sampling_type
    ds.simulation_type = simulation_type
    ds.brlen_scaler = brlen_scaler
    ds.timeout = timeout
    ds.num_start_networks = num_start_networks
    ds.fixed_reticulation_prob = fixed_reticulation_prob

    ds.n_trees = 2 ** n_reticulations
    if sampling_type == SamplingType.STANDARD:
        ds.n_trees *= m
    ds.sites_per_tree = math.ceil(approx_msa_size / ds.n_trees)
    ds.msa_size = approx_msa_size

    if sampling_type == SamplingType.STANDARD:
        ds.msa_size = ds.sites_per_tree * ds.n_trees
    if sampling_type == SamplingType.SINGLE_SITE_SAMPLING:
        ds.sites_per_tree = 1
        ds.n_trees = approx_msa_size

    for likelihood_type in likelihood_types:
        for brlen_linkage_type in brlen_linkage_types:
            for start_type in start_types:
                for use_part_msa in part_msa_types:
                    var = InferenceVariant(
                        likelihood_type, brlen_linkage_type, start_type, name, use_part_msa)
                    if start_type == StartType.RANDOM:
                        var.n_random_start_networks = num_start_networks
                        var.n_parsimony_start_networks = num_start_networks
                    elif start_type == StartType.ENDLESS:
                        var.timeout = timeout
                    ds.inference_variants.append(var)

    return ds


def sample_trees(ds, trees_prob):
    n_displayed_trees = len(trees_prob)
    sampled_trees_contrib = [0] * n_displayed_trees

    if ds.sampling_type == SamplingType.PERFECT_UNIFORM_SAMPLING:
        spt = math.ceil(ds.msa_size / n_displayed_trees)
        ds.msa_size = spt * n_displayed_trees
        sampled_trees_contrib = [spt] * n_displayed_trees
    elif ds.sampling_type == SamplingType.PERFECT_SAMPLING:
        sampled_trees_contrib = [
            math.ceil(p * ds.msa_size) for p in trees_prob]
        ds.msa_size = sum(sampled_trees_contrib)
    else:  # standard sampling and single site sampling are actually the same, just the number of sampled sites per tree differs
        tree_indices = [i for i in range(n_displayed_trees)]
        for _ in range(ds.n_trees):
            tree_idx = random.choices(
                population=tree_indices, weights=trees_prob, k=1)[0]
            sampled_trees_contrib[tree_idx] += ds.sites_per_tree

    return ds, sampled_trees_contrib


# build the trees file required by seq-gen
def build_trees_file(ds, trees_newick, sampled_trees_contrib):
    trees_file = open(ds.extracted_trees_path, "w")
    partitions_file = open(ds.partitions_path, "w")
    min_site = 1
    for i in range(len(trees_newick)):
        newick = trees_newick[i]
        if sampled_trees_contrib[i] > 0:
            trees_file.write(
                '[' + str(sampled_trees_contrib[i]) + ']' + newick + '\n')
            max_site = min_site + sampled_trees_contrib[i] - 1
            partitions_file.write(
                'DNA, tree_' + str(i) + '=' + str(min_site) + '-' + str(max_site) + '\n')
            min_site = max_site + 1
    trees_file.close()
    partitions_file.close()


#def build_dataset(n_taxa, n_reticulations, approx_msa_size, sampling_type, simulation_type, brlen_scaler, likelihood_types, brlen_linkage_types, start_types, name, timeout=0, m=1, num_start_networks=5):
#    ds = create_dataset_container(n_taxa, n_reticulations, approx_msa_size, sampling_type, simulation_type, brlen_scaler,
#                                  likelihood_types, brlen_linkage_types, start_types, name, timeout, m, num_start_networks)
#    if simulation_type == SimulatorType.SARAH:
#        generate_random_network(
#           ds.n_taxa, ds.n_reticulations, ds.true_network_path)
#    else:
#        celine_params = CelineParams()
#        celine_params.wanted_taxa = n_taxa
#        celine_params.wanted_reticulations = n_reticulations
#        ds.celine_params = simulate_network_celine(
#            ds.n_taxa, ds.n_reticulations, ds.true_network_path)
#    trees_newick, trees_prob = extract_displayed_trees(
#        ds.true_network_path, ds.n_taxa)
#    ds, sampled_trees_contrib = sample_trees(ds, trees_prob)
#    build_trees_file(ds, trees_newick, sampled_trees_contrib)
#    n_pairs, ds.n_equal_tree_pairs = check_weird_network(ds.true_network_path, ds.n_taxa)
#    if n_pairs > 0:
#        ds.true_network_weirdness = float(ds.n_equal_tree_pairs) / n_pairs
#    simulate_msa(ds)
#    return ds

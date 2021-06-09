from netrax_wrapper import *
from raxml_wrapper import *


def infer_species(tree):
    species = None
    return species


def infer_ancestral_sequences(tree, species):
    ancestral_sequences = None
    return ancestral_sequences


def collapse_species(msa, best_tree, species, species_ancestor_sequences):
    reduced_msa = None
    reduced_tree = None
    return reduced_msa, reduced_tree


def uncollapse_species_subtrees(reduced_network, orig_tree, species):
    uncollapsed_network = None
    return uncollapsed_network


def infer_network_netrax(msa, partitions, start_tree):
    inferred_network = None
    return inferred_network


def infer_tree_netrax(msa, start_tree=None, n_random=10, n_parsimony=10):
    inferred_tree = None
    return inferred_tree


def subsampled_inference_run(start_tree, full_msa, partitions, is_best_tree):
    if not is_best_tree or start_tree == None:
        best_tree = infer_tree_netrax(full_msa, partitions, start_tree)
    else:
        best_tree = start_tree
    species = infer_species(best_tree)
    species_ancestor_sequences = infer_ancestral_sequences(best_tree, species)
    reduced_msa, reduced_tree = collapse_species(full_msa, best_tree, species_ancestor_sequences)
    reduced_network = infer_network_netrax(reduced_msa, partitions, reduced_tree)
    uncollapsed_network = uncollapse_species_subtrees(reduced_network, best_tree, species)
    final_network = infer_network_netrax(full_msa, partitions, uncollapsed_network)
    return final_network


if __name__== "__main__":
    start_tree = None
    full_msa = None
    partitions = None
    is_best_tree = None
    final_network = subsampled_inference_run(start_tree, full_msa, partitions, is_best_tree)
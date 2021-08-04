from phylinc_wrapper import run_phylinc_inference
from experiment_settings import exp_standard
from netrax_wrapper import infer_networks, check_weird_network, extract_displayed_trees, change_reticulation_prob_only
from evaluate_experiments import run_inference_and_evaluate, write_results_to_csv
from run_experiments import simulate_network_celine_fixed_nonweird, build_dataset
from dataset_builder import sample_trees, build_trees_file
from seqgen_wrapper import simulate_msa
from append_topological_distances import append_distances_netrax
from append_msa_patterns import append_patterns
from csv_merger import postprocess_merge
import argparse
import subprocess
import copy

from raxml_wrapper import infer_raxml_tree, compute_rf_dist

def parse_command_line_arguments_phylinc_experiment():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="phylinc_exp")
    CLI.add_argument("--n_taxa", type=int, default=10)
    CLI.add_argument("--n_reticulations", type=int, default=1)
    args = CLI.parse_args()
    return args.prefix, args.n_taxa, args.n_reticulations


def simulate_stuff(prefix, n_taxa, n_reticulations):
    (_, settings) = exp_standard(n_taxa, n_reticulations)
    settings.folder_path = 'data/'
    _, _, newick, param_info = simulate_network_celine_fixed_nonweird(settings, n_taxa, n_reticulations)
    n_trees = 2 ** param_info["no_of_hybrids"]
    ds = build_dataset(prefix, n_taxa, n_reticulations, n_trees * settings.partition_sizes[0], settings.simulator_types[0], settings.fixed_brlen_scalers[0],
                  settings.sampling_types[0], settings.likelihood_types, settings.brlen_linkage_types, settings.start_types, settings.use_partitioned_msa_types, settings.folder_path, 0, 0, 0.5)
    ds.sites_per_tree = settings.partition_sizes[0]
    ds.celine_params = param_info
    ds.n_trees = 2 ** ds.celine_params["no_of_hybrids"]

    network_file = open(ds.true_network_path, "w")
    network_file.write(newick + '\n')
    network_file.close()
    change_reticulation_prob_only(ds.true_network_path, ds.true_network_path, 0.5, n_taxa)
                                    
    # network topology has been simulated now.
    n_pairs, ds.n_equal_tree_pairs = check_weird_network(ds.true_network_path, ds.n_taxa)
    if n_pairs > 0:
        ds.true_network_weirdness = float(ds.n_equal_tree_pairs) / n_pairs
    trees_newick, trees_prob = extract_displayed_trees(ds.true_network_path, ds.n_taxa)
    if len(trees_newick) < 2 ** n_reticulations:
        raise Exception("something went wrong when extracting displayed trees")
    ds, sampled_trees_contrib = sample_trees(ds, trees_prob)
    build_trees_file(ds, trees_newick, sampled_trees_contrib)
    simulate_msa(ds)
    return ds


def to_fasta(phylip_path, fasta_path):
    with open(phylip_path) as f:
        lines = f.readlines()[1:]
    outfile = open(fasta_path, 'w')
    for line in lines:
        if "        " in line:
            outfile.write(">" + line.split("        ")[0].strip() + "\n" + line.split("        ")[1].strip() + "\n")
    outfile.close()


if __name__ == '__main__':
    prefix, n_taxa, n_reticulations = parse_command_line_arguments_phylinc_experiment()
    ds = simulate_stuff(prefix, n_taxa, n_reticulations)
    ds.near_zero_branches_raxml = infer_raxml_tree(ds)
    infile_name = ds.partitions_path
    
    to_fasta(ds.msa_path, ds.msa_path + ".fasta")

    phylinc_network, phylinc_runtime = run_phylinc_inference(ds.name + ".raxml.bestTree", ds.msa_path + ".fasta", n_reticulations+1)
    print(phylinc_network)
    print(str(phylinc_runtime) + " seconds")
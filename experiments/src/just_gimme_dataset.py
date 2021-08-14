from scramble_partitions import scramble_partitions, write_partitions
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
import copy


def parse_command_line_arguments_gimme():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="dataset")
    CLI.add_argument("--n_taxa", type=int, default=30)
    CLI.add_argument("--n_reticulations", type=int, default=3)
    CLI.add_argument("--partition_size", type=int, default=1000)
    CLI.add_argument("--min_reticulation_prob", type=float, default=0.5)
    CLI.add_argument("--max_reticulation_prob", type=float, default=0.5)
    args = CLI.parse_args()
    return args.prefix, args.n_taxa, args.n_reticulations, args.partition_size, args.min_reticulation_prob, args.max_reticulation_prob


def simulate_my_stuff(prefix, n_taxa, n_reticulations, partition_size, min_reticulation_prob, max_reticulation_prob):
    (_, settings) = exp_standard(n_taxa, n_reticulations)
    settings.folder_path = 'data/'
    settings.partition_sizes = [partition_size]
    settings.min_reticulation_prob = min_reticulation_prob
    settings.max_reticulation_prob = max_reticulation_prob
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


if __name__ == '__main__':
    prefix, n_taxa, n_reticulations, partition_size, min_reticulation_prob, max_reticulation_prob = parse_command_line_arguments_gimme()
    simulate_my_stuff(prefix, n_taxa, n_reticulations, partition_size, min_reticulation_prob, max_reticulation_prob)

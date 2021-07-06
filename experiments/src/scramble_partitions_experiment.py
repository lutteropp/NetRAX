from scramble_partitions import scramble_partitions, write_partitions
from experiment_settings import exp_standard
from netrax_wrapper import infer_networks
from evaluate_experiments import run_inference_and_evaluate, write_results_to_csv
import argparse
import copy


SCRAMBLE_FACTOR=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def parse_command_line_arguments_pscramble_exp():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="dataset")
    CLI.add_argument("--n_taxa", type=int, default=40)
    CLI.add_argument("--n_reticulations", type=int, default=4)
    args = CLI.parse_args()
    return args.prefix, args.n_taxa, args.n_reticulations


def simulate_stuff(prefix, n_taxa, n_reticulations):
    (_, settings) = exp_standard(n_taxa, n_reticulations)
    _, _, newick, param_info = simulate_network_celine_fixed_nonweird(settings, n_taxa, n_reticulations)
    n_trees = 2 ** param_info["no_of_hybrids"]
    ds = build_dataset(prefix, n_taxa, n_reticulations, n_trees * settings.partition_sizes[0], settings.simulator_types[0], settings.fixed_brlen_scalers[0],
                  settings.sampling_types[0], settings.likelihood_types, settings.brlen_linkage_types, settings.start_types, settings.use_partitioned_msa_types, settings.folder_path, 0, 0, 0.5)
    ds.sites_per_tree = partition_size
    ds.celine_params = param_info
    ds.n_trees = 2 ** ds.celine_params["no_of_hybrids"]

    network_file = open(ds.true_network_path, "w")
    network_file.write(newick + '\n')
    network_file.close()
                                    
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
    prefix, n_taxa, n_reticulations = parse_command_line_arguments_pscramble_exp()
    ds = simulate_stuff(prefix, n_taxa, n_reticulations)
    infile_name = ds.partitions_path
    
    new_datasets = []
    for fraction in SCRAMBLE_FACTOR:
        outfile_name = infile_name.split('.')[0] + '_' + str(fraction).replace('.', '_') + '.txt'
        model, name, new_psites = scramble_partitions(infile_name, outfile_name, fraction)
        write_partitions(model, name, new_psites, outfile_name)
        new_ds = copy.deepcopy(ds)
        new_ds.partitions_path = outfile_name
        new_datasets.append(new_ds)

    for new_ds in new_datasets:
        infer_networks(new_ds)

    run_inference_and_evaluate(new_datasets)
    write_results_to_csv(new_datasets, settings.folder_path + prefix + "_results.csv")
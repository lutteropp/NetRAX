import os
import random

from dataset_builder import create_dataset_container, sample_trees, build_trees_file
from netrax_wrapper import extract_displayed_trees, check_weird_network, scale_branches_only_newick, change_reticulation_prob_only
from evaluate_experiments import SamplingType, SimulatorType, LikelihoodType, BrlenLinkageType, StartType, run_inference_and_evaluate, write_results_to_csv
from seqgen_wrapper import simulate_msa

from celine_simulator import simulate_network_celine_minmax, simulate_network_celine_fixed

import argparse

from experiment_settings import ExperimentSettings, gather_labeled_settings


def build_dataset(prefix, n_taxa, n_reticulations, msa_size, simulator_type, brlen_scaler, sampling_type, likelihood_types, brlen_linkage_types, start_types, part_msa_types, folder_path, my_id, reticulation_prob=None):
    if not os.path.exists(folder_path + 'datasets_' + prefix):
        os.makedirs(folder_path + 'datasets_' + prefix)
    name = folder_path + "datasets_" + prefix + "/" + str(my_id) + '_' + str(n_taxa) + '_taxa_' + str(
        n_reticulations) + '_reticulations_' + str(simulator_type) + "_" + str(sampling_type) + "_" + str(msa_size) + "_msasize" + "_" + str(brlen_scaler).replace(".","_") + "_brlenScaler"
    if reticulation_prob:
        name += str(reticulation_prob).replace('.','_') + "_reticulation_prob"
    return create_dataset_container(n_taxa, n_reticulations, msa_size, sampling_type, simulator_type, brlen_scaler, likelihood_types, brlen_linkage_types, start_types, part_msa_types, my_id, name, reticulation_prob)


def simulate_network_celine_minmax_nonweird(settings):
    temp_path = "temp_network_" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".txt"
    n_taxa, n_reticulations, newick, param_info = simulate_network_celine_minmax(
                settings.min_taxa, settings.max_taxa, settings.min_reticulations, settings.max_reticulations, settings.min_reticulation_prob, settings.max_reticulation_prob)
   
    network_file = open(temp_path, "w")
    network_file.write(newick + '\n')
    network_file.close()

    _, n_equal = check_weird_network(temp_path, n_taxa)
    os.remove(temp_path)
    if n_equal == 0:
        return n_taxa, n_reticulations, newick, param_info
    else:
        return simulate_network_celine_minmax_nonweird(settings)


def simulate_network_celine_fixed_nonweird(settings, n_taxa, n_reticulations):
    temp_path = "temp_network_" + str(os.getpid()) + "_" + str(random.getrandbits(64)) + ".txt"
    n_taxa, n_reticulations, newick, param_info = simulate_network_celine_fixed(
                n_taxa, n_reticulations, settings.min_reticulation_prob, settings.max_reticulation_prob)
   
    network_file = open(temp_path, "w")
    network_file.write(newick + '\n')
    network_file.close()

    _, n_equal = check_weird_network(temp_path, n_taxa)
    os.remove(temp_path)
    if n_equal == 0:
        return n_taxa, n_reticulations, newick, param_info
    else:
        return simulate_network_celine_fixed(
                n_taxa, n_reticulations, settings.min_reticulation_prob, settings.max_reticulation_prob)


def simulate_datasets_range(prefix, settings, iterations):
    datasets = []
    for my_id in range(iterations):
        n_taxa, n_reticulations, newick, param_info = simulate_network_celine_minmax_nonweird(settings)
        n_trees = 2 ** param_info["no_of_hybrids"]
        for partition_size in settings.partition_sizes:
            for sampling_type in settings.sampling_types:
                ds = build_dataset(prefix, n_taxa, n_reticulations, n_trees * partition_size, SimulatorType.CELINE, 1,
                                sampling_type, settings.likelihood_types, settings.brlen_linkage_types, settings.start_types, settings.use_partitioned_msa_types, settings.folder_path, my_id)
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
                trees_newick, trees_prob = extract_displayed_trees(
                    ds.true_network_path, ds.n_taxa)
                ds, sampled_trees_contrib = sample_trees(ds, trees_prob)
                build_trees_file(ds, trees_newick, sampled_trees_contrib)
                simulate_msa(ds)
                datasets.append(ds)
    return datasets


def simulate_datasets_fixed(prefix, settings, iterations):
    datasets = []
    for my_id in range(iterations):
        for simulator_type in settings.simulator_types:   
            for n_taxa in settings.fixed_n_taxa:
                for n_reticulations in settings.fixed_n_reticulations:
                    n_taxa, n_reticulations, newick, param_info = simulate_network_celine_fixed_nonweird(settings, n_taxa, n_reticulations)
                    n_trees = 2 ** param_info["no_of_hybrids"]
                    for brlen_scaler in settings.fixed_brlen_scalers:
                        for reticulation_prob in settings.fixed_reticulation_probs:
                            for partition_size in settings.partition_sizes:
                                for sampling_type in settings.sampling_types:
                                    ds = build_dataset(prefix, n_taxa, n_reticulations, n_trees * partition_size, simulator_type, brlen_scaler,
                                                    sampling_type, settings.likelihood_types, settings.brlen_linkage_types, settings.start_types, settings.use_partitioned_msa_types, settings.folder_path, my_id, reticulation_prob)
                                    ds.sites_per_tree = partition_size
                                    ds.celine_params = param_info
                                    ds.n_trees = 2 ** ds.celine_params["no_of_hybrids"]

                                    if brlen_scaler != 1.0:
                                        scale_branches_only_newick(newick, ds.true_network_path, brlen_scaler, n_taxa)
                                    else:
                                        network_file = open(ds.true_network_path, "w")
                                        network_file.write(newick + '\n')
                                        network_file.close()

                                    change_reticulation_prob_only(ds.true_network_path, ds.true_network_path, reticulation_prob, n_taxa)

                                    # network topology has been simulated now.
                                    n_pairs, ds.n_equal_tree_pairs = check_weird_network(ds.true_network_path, ds.n_taxa)
                                    if n_pairs > 0:
                                        ds.true_network_weirdness = float(ds.n_equal_tree_pairs) / n_pairs
                                    trees_newick, trees_prob = extract_displayed_trees(
                                        ds.true_network_path, ds.n_taxa)
                                    ds, sampled_trees_contrib = sample_trees(ds, trees_prob)
                                    build_trees_file(ds, trees_newick, sampled_trees_contrib)
                                    simulate_msa(ds)
                                    datasets.append(ds)
    return datasets


def simulate_datasets(prefix, settings, iterations):
    if not os.path.exists(settings.folder_path + 'datasets_' + prefix):
        os.makedirs(settings.folder_path + 'datasets_' + prefix)

    if SimulatorType.SARAH in settings.simulator_types:
        raise Exception("Only SimulatorType.CELINE please")

    if settings.use_fixed_simulation:
        print("Using fixed simulation")
        return simulate_datasets_fixed(prefix, settings, iterations)
    else:
        print("Using range simulation")
        return simulate_datasets_range(prefix, settings, iterations)


def run_experiments(prefix, settings, iterations):
    if settings.folder_path != "" and not os.path.isdir(settings.folder_path):
        os.makedirs(settings.folder_path)
    datasets = simulate_datasets(prefix, settings, iterations)
    print("Simulated " + str(len(datasets)) + " datasets.")
    run_inference_and_evaluate(datasets)
    write_results_to_csv(datasets, settings.folder_path + prefix + "_results.csv")


def parse_command_line_arguments_experiment():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="small_network")
    CLI.add_argument("--sampling_types", nargs="*",
                     type=SamplingType, default=[SamplingType.PERFECT_SAMPLING])
    CLI.add_argument("--start_types", nargs="*", type=StartType,
                     default=[StartType.FROM_RAXML, StartType.RANDOM])
    CLI.add_argument("--simulator_types", nargs="*", type=SimulatorType, default=[SimulatorType.CELINE])
    CLI.add_argument("--brlen_linkage_types", nargs="*",
                     type=BrlenLinkageType, default=[BrlenLinkageType.LINKED])
    CLI.add_argument("--likelihood_types", nargs="*", type=LikelihoodType,
                     default=[LikelihoodType.BEST, LikelihoodType.AVERAGE])
    CLI.add_argument("--partition_sizes", nargs="*",
                     type=int, default=[500, 1000])
    CLI.add_argument("--brlen_scalers", nargs="*", type=float, default=[1.0])
    CLI.add_argument("--min_taxa", type=int, default=4)
    CLI.add_argument("--max_taxa", type=int, default=10)
    CLI.add_argument("--min_reticulations", type=int, default=1)
    CLI.add_argument("--max_reticulations", type=int, default=2)
    CLI.add_argument("--min_reticulation_prob", type=float, default=0.1)
    CLI.add_argument("--max_reticulation_prob", type=float, default=0.9)
    CLI.add_argument("--folder_path", type=str, default="data/")

    CLI.add_argument("--labeled_settings", type=str, default="")

    args = CLI.parse_args()

    known_setups = gather_labeled_settings()
    if len(args.labeled_settings) > 0:
        if args.labeled_settings in known_setups:
            _, settings = known_setups[args.labeled_settings]
            settings.folder_path = args.folder_path
            return args.prefix, settings
        else:
            raise Exception("Unknown experiment name: " + args.labeled_settings)
    else:
        settings = ExperimentSettings()
        settings.sampling_types = args.sampling_types
        settings.start_types = args.start_types
        settings.simulator_types = args.simulator_types
        settings.brlen_linkage_types = args.brlen_linkage_types
        settings.likelihood_types = args.likelihood_types
        settings.partition_sizes = args.partition_sizes
        settings.brlen_scalers = args.brlen_scalers
        settings.min_taxa = args.min_taxa
        settings.max_taxa = args.max_taxa
        settings.min_reticulations = args.min_reticulations
        settings.max_reticulations = args.max_reticulations
        settings.min_reticulation_prob = args.min_reticulation_prob
        settings.max_reticulation_prob = args.max_reticulation_prob
        settings.folder_path = args.folder_path
        return args.prefix, settings


if __name__ == "__main__":
    prefix, settings = parse_command_line_arguments_experiment()
    run_experiments(prefix, settings, 1)

from scramble_partitions import scramble_partitions, write_partitions
from experiment_settings import exp_runtime
from netrax_wrapper import infer_networks, check_weird_network, extract_displayed_trees
from raxml_wrapper import infer_raxml_tree
from evaluate_experiments import run_inference_and_evaluate, write_results_to_csv
from run_experiments import simulate_network_celine_fixed_nonweird, build_dataset
from dataset_builder import sample_trees, build_trees_file
from seqgen_wrapper import simulate_msa
from append_topological_distances import append_distances_netrax
from append_msa_patterns import append_patterns
from csv_merger import postprocess_merge
import argparse
import copy


MPI_PROCS = [64, 32, 16, 8, 4, 2, 1]

def parse_command_line_arguments_mpi_exp():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="mpi_runtime")
    CLI.add_argument("--n_taxa", type=int, default=20)
    CLI.add_argument("--n_reticulations", type=int, default=3)
    CLI.add_argument("--iteration", type=int, default=0)
    args = CLI.parse_args()
    prefix = args.prefix + '_' + str(args.iteration)
    return prefix, args.n_taxa, args.n_reticulations, args.iteration


def simulate_stuff(prefix, n_taxa, n_reticulations):
    (_, settings) = exp_runtime(n_taxa, n_reticulations)
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
    prefix, n_taxa, n_reticulations, iteration = parse_command_line_arguments_mpi_exp()
    ds = simulate_stuff(prefix, n_taxa, n_reticulations)
    ds.near_zero_branches_raxml = infer_raxml_tree(ds)
    infile_name = ds.partitions_path
    
    act_id = iteration
    new_datasets = []
    for procs in MPI_PROCS:
        new_ds = copy.deepcopy(ds)
        new_ds.my_id = act_id
        name = 'data/' + "datasets_" + prefix + "/" + str(procs) + '_' + str(new_ds.my_id)
        new_ds.name = name
        for var in new_ds.inference_variants:
            var.inferred_network_path = name + "_" + str(var.likelihood_type) + "_" + str(
            var.brlen_linkage_type) + "_" + str(var.start_type) + "_inferred_network.nw"
        new_ds.mpi_procs = procs
        infer_networks(new_ds, procs)
        evaluate_dataset(new_ds)
        write_results_to_csv(new_datasets, 'data/' + prefix + '_' + str(act_id) + "_procs_" + str(procs) + "_intermediate_results.csv")
        new_datasets.append(new_ds)

    write_results_to_csv(new_datasets, 'data/' + prefix + '_' + str(act_id) + "_results.csv")

    append_patterns(prefix + '_' + str(act_id))
    append_distances_netrax(prefix + '_' + str(act_id))

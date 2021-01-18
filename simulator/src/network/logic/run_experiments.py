import os
from dataset_builder import create_dataset_container, sample_trees, build_trees_file
from netrax_wrapper import extract_displayed_trees
from collections import defaultdict
from evaluate_experiments import SamplingType, SimulatorType, LikelihoodType, BrlenLinkageType, StartType, run_inference_and_evaluate, write_results_to_csv
from seqgen_wrapper import simulate_msa

from celine_simulator import CelineParams, simulate_network_celine_minmax

from create_plots import create_plots

def build_dataset(prefix, n_taxa, n_reticulations, msa_size, simulator_type, sampling_type, likelihood_types, brlen_linkage_types, start_types, my_id):
    if not os.path.exists('datasets_' + prefix):
        os.makedirs('datasets_' + prefix)
    name = "datasets_" + prefix + "/" + str(my_id) + '_' + str(n_taxa) + '_taxa_' + str(n_reticulations) + '_reticulations_' + str(simulator_type) + "_" + str(sampling_type) + "_" + str(msa_size) + "_msasize"
    return create_dataset_container(n_taxa, n_reticulations, msa_size, sampling_type, simulator_type, likelihood_types, brlen_linkage_types, start_types, my_id, name)
    

def run_experiments(prefix, iterations, min_taxa, max_taxa, min_reticulations, max_reticulations, sampling_types, likelihood_types, brlen_linkage_types, start_types, partition_sizes):
    if not os.path.exists('datasets_' + prefix):
        os.makedirs('datasets_' + prefix)
    datasets = []
    results = []
    
    counter = dict()
    for taxa in range(max_taxa+1):
        counter[taxa] = defaultdict(int)
        
    for my_id in range(iterations):
        n_taxa, n_reticulations, newick, param_info = simulate_network_celine_minmax(min_taxa, max_taxa, min_reticulations, max_reticulations)
        counter[n_taxa][n_reticulations] += 1
        n_trees = 2 ** param_info["no_of_hybrids"]
        for partition_size in partition_sizes:
            for sampling_type in sampling_types:
                ds = build_dataset(prefix, n_taxa, n_reticulations, n_trees * partition_size, SimulatorType.CELINE, sampling_type, likelihood_types, brlen_linkage_types, start_types, my_id)
                ds.sites_per_tree = partition_size
                ds.celine_params = param_info
                ds.n_trees = 2 ** ds.celine_params["no_of_hybrids"]
                network_file = open(ds.true_network_path, "w")
                network_file.write(newick + '\n')
                network_file.close()
                # network topology has been simulated now.
                trees_newick, trees_prob = extract_displayed_trees(ds.true_network_path, ds.n_taxa)
                ds, sampled_trees_contrib = sample_trees(ds, trees_prob)
                build_trees_file(ds, trees_newick, sampled_trees_contrib)
                simulate_msa(ds)
                datasets.append(ds)
    
    for i in range(max_taxa+1):
        for j in range(max_reticulations+1):
            if counter[i][j] != 0:
                print(str(i) + ", " + str(j) + ": " + str(counter[i][j]))
    
    run_inference_and_evaluate(datasets)
    write_results_to_csv(datasets, prefix + "_results.csv")
    create_plots(prefix)


def run_experiments_small_tree():
    iterations = 100
    min_taxa = 4
    max_taxa = 10
    min_reticulations = 0
    max_reticulations = 0
    prefix = 'small_tree'
    sampling_types = [SamplingType.PERFECT_SAMPLING]
    start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    brlen_linkage_types = [BrlenLinkageType.SCALED]
    likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    partition_sizes = [500, 1000]
    run_experiments(prefix, iterations, min_taxa, max_taxa, min_reticulations, max_reticulations, sampling_types, likelihood_types, brlen_linkage_types, start_types, partition_sizes)
    

def run_experiments_small_network():
    iterations = 10
    min_taxa = 4
    max_taxa = 10
    min_reticulations = 1
    max_reticulations = 2
    prefix = 'small_network'
    sampling_types = [SamplingType.PERFECT_SAMPLING]
    start_types = [StartType.FROM_RAXML]
    brlen_linkage_types = [BrlenLinkageType.LINKED, BrlenLinkageType.SCALED, BrlenLinkageType.UNLINKED]
    likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    partition_sizes = [500, 1000]
    run_experiments(prefix, iterations, min_taxa, max_taxa, min_reticulations, max_reticulations, sampling_types, likelihood_types, brlen_linkage_types, start_types, partition_sizes)
    
    
def run_experiments_small_network_single_debug():
    iterations = 1
    min_taxa = 4
    max_taxa = 4
    min_reticulations = 1
    max_reticulations = 1
    prefix = 'small_network_single_debug'
    sampling_types = [SamplingType.PERFECT_SAMPLING]
    start_types = [StartType.FROM_RAXML]
    brlen_linkage_types = [BrlenLinkageType.SCALED]
    likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    partition_sizes = [1000]
    run_experiments(prefix, iterations, min_taxa, max_taxa, min_reticulations, max_reticulations, sampling_types, likelihood_types, brlen_linkage_types, start_types, partition_sizes)
    
    
def run_experiments_larger_network():
    iterations = 1
    min_taxa = 20
    max_taxa = 50
    min_reticulations = 2
    max_reticulations = -1
    prefix = 'larger_network'
    sampling_types = [SamplingType.PERFECT_SAMPLING]
    start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    brlen_linkage_types = [BrlenLinkageType.SCALED]
    likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    partition_sizes = [500, 1000]
    run_experiments(prefix, iterations, min_taxa, max_taxa, min_reticulations, max_reticulations, sampling_types, likelihood_types, brlen_linkage_types, start_types, partition_sizes)
    
    
if __name__ == "__main__":
    #run_experiments_small_tree()
    run_experiments_small_network()
    #run_experiments_larger_network()
    #run_experiments_small_network_single_debug()

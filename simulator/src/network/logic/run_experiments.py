import os
from dataset_builder import create_dataset_container, sample_trees, build_trees_file
from netrax_wrapper import extract_displayed_trees
from collections import defaultdict
from evaluate_experiments import SamplingType, SimulatorType, LikelihoodType, BrlenLinkageType, StartType, run_inference_and_evaluate, write_results_to_csv
from seqgen_wrapper import simulate_msa

from celine_simulator import CelineParams, simulate_network_celine_minmax

from create_plots import create_plots

class ExperimentSettings:
    def __init__(self):
        self.iterations = 1
        self.min_taxa = 4
        self.max_taxa = 10
        self.min_reticulations = 1
        self.max_reticulations = 2
        self.sampling_types = [SamplingType.PERFECT_SAMPLING]
        self.simulator_types = [SimulatorType.CELINE]
        self.likelihood_types = [LikelihoodType.AVERAGE, LikelihoodType.BEST]
        self.brlen_linkage_types = [BrlenLinkageType.LINKED]
        self.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
        self.partition_sizes = [1000]


def build_dataset(prefix, n_taxa, n_reticulations, msa_size, simulator_type, sampling_type, likelihood_types, brlen_linkage_types, start_types, my_id):
    if not os.path.exists('datasets_' + prefix):
        os.makedirs('datasets_' + prefix)
    name = "datasets_" + prefix + "/" + str(my_id) + '_' + str(n_taxa) + '_taxa_' + str(n_reticulations) + '_reticulations_' + str(simulator_type) + "_" + str(sampling_type) + "_" + str(msa_size) + "_msasize"
    return create_dataset_container(n_taxa, n_reticulations, msa_size, sampling_type, simulator_type, likelihood_types, brlen_linkage_types, start_types, my_id, name)
    

def simulate_datasets(prefix, settings):
    if not os.path.exists('datasets_' + prefix):
        os.makedirs('datasets_' + prefix)

    if SimulatorType.SARAH in settings.simulator_types:
        raise Exception("Only SimulatorType.CELINE please")
    
    counter = dict()
    for taxa in range(settings.max_taxa+1):
        counter[taxa] = defaultdict(int)
        
    datasets = []
    for my_id in range(settings.iterations):
        n_taxa, n_reticulations, newick, param_info = simulate_network_celine_minmax(settings.min_taxa, settings.max_taxa, settings.min_reticulations, settings.max_reticulations)
        counter[n_taxa][n_reticulations] += 1
        n_trees = 2 ** param_info["no_of_hybrids"]
        for partition_size in settings.partition_sizes:
            for sampling_type in settings.sampling_types:
                ds = build_dataset(prefix, n_taxa, n_reticulations, n_trees * partition_size, SimulatorType.CELINE, sampling_type, settings.likelihood_types, settings.brlen_linkage_types, settings.start_types, my_id)
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
    
    print("Statistics on simulated datasets (n_taxa, n_reticulations, count):")
    for i in range(settings.max_taxa+1):
        for j in range(settings.max_reticulations+1):
            if counter[i][j] != 0:
                print(str(i) + ", " + str(j) + ": " + str(counter[i][j]))

    return datasets


def run_experiments(prefix, settings):
    datasets = simulate_datasets(prefix, settings)
    run_inference_and_evaluate(datasets)
    write_results_to_csv(datasets, prefix + "_results.csv")
    create_plots(prefix)


def run_experiments_small_tree():
    settings = ExperimentSettings()
    prefix = 'small_tree'
    settings.iterations = 1
    settings.min_taxa = 4
    settings.max_taxa = 10
    settings.min_reticulations = 0
    settings.max_reticulations = 0
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.SCALED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [500, 1000]
    run_experiments(prefix, settings)
    

def run_experiments_small_network():
    settings = ExperimentSettings()
    prefix = 'small_network'
    settings.iterations = 1
    settings.min_taxa = 4
    settings.max_taxa = 10
    settings.min_reticulations = 1
    settings.max_reticulations = 2
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [500, 1000]
    run_experiments(prefix, settings)
    

def run_experiments_small_network_single_debug(): 
    settings = ExperimentSettings()
    prefix = 'small_network_single_debug'
    settings.iterations = 1
    settings.min_taxa = 4
    settings.max_taxa = 4
    settings.min_reticulations = 1
    settings.max_reticulations = 1
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    run_experiments(prefix, settings)
    

def run_experiments_larger_network():
    settings = ExperimentSettings()
    prefix = 'larger_network'
    settings.iterations = 1
    settings.min_taxa = 20
    settings.max_taxa = 50
    settings.min_reticulations = 2
    settings.max_reticulations = -1
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [500, 1000]
    run_experiments(prefix, settings)
  
    
if __name__ == "__main__":
    #run_experiments_small_tree()
    run_experiments_small_network()
    #run_experiments_larger_network()
    #run_experiments_small_network_single_debug()

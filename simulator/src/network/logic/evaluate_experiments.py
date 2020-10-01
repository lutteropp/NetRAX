from dendroscope_wrapper import *
from netrax_wrapper import *
from enum import Enum
    
    
class SamplingType(Enum):
    STANDARD = 1 # randomly choose which tree to sample, then sample equal number of sites for each sampled tree - this is the only mode that uses the n_trees or m parameter for sampling
    PERFECT_SAMPLING = 2 # sample each displayed tree, and as many site as expected by the tree probability
    PERFECT_UNIFORM_SAMPLING = 3 # sample each displayed tree, with the same number of sites per tree (ignoring reticulation probabilities)
    SINGLE_SITE_SAMPLING = 4 # sample each site individually, with the reticulation probabilities in mind
    
    
class SimulationType(Enum):
    CELINE = 1 # use Celine's network topology simulator
    SARAH = 2 # use Sarah's ad-hoc network topology generator
    
    

TOPOLOGICAL_DISTANCE_NAMES = ['hardwired_cluster_distance', 'softwired_cluster_distance', 'displayed_trees_distance', 'tripartition_distance', 'nested_labels_distance', 'path_multiplicity_distance']

DATASET_CSV_HEADER = "n_taxa,n_trees,n_reticulations,msa_size,sampling_type,simulation_type,timeout"
RESULT_CSV_HEADER = "n_reticulations_inferred,bic_true,logl_true,bic_inferred,logl_inferred" + "," + ",".join(TOPOLOGICAL_DISTANCE_NAMES)


class Dataset:
    def __init__(self):
        self.n_taxa = 0
        self.n_reticulations = 0
        self.n_trees = 0
        self.sites_per_tree = 0
        self.msa_path = ""
        self.extracted_trees_path = ""
        self.true_network_path = ""
        self.inferred_network_path = ""
        self.sampling_type = SamplingType.STANDARD
        self.simulation_type = SimulationType.CELINE
        self.timeout = 0
        
    def msa_size(self):
        return self.n_trees * self.sites_per_tree
        
    def get_csv_line(self):
        return str(self.n_taxa) + "," + str(self.n_trees) + "," + str(self.n_reticulations) + "," + str(self.msa_size) + "," + str(self.sampling_type) + "," + str(self.simulation_type) + "," + str(self.timeout)


class Result:
    def __init__(self, dataset):
        self.dataset = dataset
        self.bic_true = 0
        self.logl_true = 0
        self.bic_inferred = 0
        self.logl_inferred = 0
        self.n_reticulations_inferred = 0
        self.topological_distances = {}
        
    def get_csv_line(self):
        topo_scores_strings_ordered = [str(self.topological_distances[x]) for x in TOPOLOGICAL_DISTANCE_NAMES]
        return str(self.n_reticulations_inferred) + "," + str(self.bic_true) + "," + str(self.logl_true) + "," + str(self.bic_inferred) + "," + str(self.logl_inferred) + ",".join(topo_scores_strings_ordered)
        

def evaluate_dataset(dataset):
    res = Result(dataset)
    _, res.bic_true, res.logl_true = score_network(datase.true_network_path, dataset.msa_path)
    res.n_reticulations_inferred, res.bic_inferred, res.logl_inferred = score_network(datase.inferred_network_path, dataset.msa_path)
    network_1 = open(dataset.true_network_path).read()
    network_2 = open(dataset.inferred_network_path).read()
    res.topological_distances = get_dendro_scores(network_1, network_2)
    print(RESULT_CSV_HEADER+"\n" + res.get_csv_line + "\n\n)
    return res
    
    
def run_inference_and_evaluate(datasets):
    for ds in datasets:
        infer_network(ds.msa_path, ds.inferred_network_path, ds.timeout)
    results = [evaluate_dataset(dataset) for ds in datasets]
    return results
    

def write_results_to_csv(results, csv_path):
    csv_file = open(csv_path, "w")
    header = DATASET_CSV_HEADER + ";" + RESULT_CSV_HEADER
    csv_file.write(header + "\n")
    for res in results:
        line = str(res.dataset.get_csv_line() + "," + res.get_csv_line() + "\n")
        csv_file.write(line)
    csv_file.close()


def run_experiments():
    pass


if __name__ == "__init__":
    run_experiments()

from enum import Enum
    
    
class SamplingType(Enum):
    STANDARD = 1 # randomly choose which tree to sample, then sample equal number of sites for each sampled tree - this is the only mode that uses the n_trees or m parameter for sampling
    PERFECT_SAMPLING = 2 # sample each displayed tree, and as many site as expected by the tree probability
    PERFECT_UNIFORM_SAMPLING = 3 # sample each displayed tree, with the same number of sites per tree (ignoring reticulation probabilities)
    SINGLE_SITE_SAMPLING = 4 # sample each site individually, with the reticulation probabilities in mind
    
    
class SimulationType(Enum):
    CELINE = 1 # use Celine's network topology simulator
    SARAH = 2 # use Sarah's ad-hoc network topology generator
    
    
class LikelihoodType(Enum):
    AVERAGE = 1 # use weighted average of displayed trees
    BEST = 2 # use best displayed tree


TOPOLOGICAL_DISTANCE_NAMES = ['hardwired_cluster_distance', 'softwired_cluster_distance', 'displayed_trees_distance', 'tripartition_distance', 'nested_labels_distance', 'path_multiplicity_distance']

DATASET_CSV_HEADER = "name,n_taxa,n_trees,n_reticulations,msa_size,sampling_type,simulation_type,likelihood_type,timeout,n_random_start_networks,n_parsimony_start_networks,start_from_raxml"
RESULT_CSV_HEADER = "n_reticulations_inferred,bic_true,logl_true,bic_inferred,logl_inferred,bic_raxml,logl_raxml,rf_absolute_raxml,rf_relative_raxml,rf_absolute_inferred,rf_relative_inferred,near_zero_branches_raxml,runtime_inference" + "," + ",".join(TOPOLOGICAL_DISTANCE_NAMES)


class Dataset:
    def __init__(self):
        self.n_taxa = 0
        self.n_reticulations = 0
        self.n_trees = 0
        self.sites_per_tree = 0
        self.msa_path = ""
        self.partitions_path = ""
        self.extracted_trees_path = ""
        self.true_network_path = ""
        self.inferred_network_path = ""
        self.inferred_network_with_raxml_path = ""
        self.raxml_tree_path = ""
        self.name = ""
        self.sampling_type = SamplingType.PERFECT_SAMPLING
        self.simulation_type = SimulationType.CELINE
        self.likelihood_type = LikelihoodType.AVERAGE
        self.timeout = 0
        self.n_random_start_networks = 10
        self.n_parsimony_start_networks = 10
        self.start_from_raxml = True
        self.celine_params = {}
        
    def msa_size(self):
        return self.n_trees * self.sites_per_tree
        
    def get_csv_line(self):
        return str(self.name) + "," + str(self.n_taxa) + "," + str(self.n_trees) + "," + str(self.n_reticulations) + "," + str(self.msa_size) + "," + str(self.sampling_type) + "," + str(self.simulation_type) + "," + str(self.likelihood_type) + "," + str(self.timeout) + "," + str(self.n_random_start_networks) + "," + str(self.n_parsimony_start_networks) + ",False"
        
    def get_csv_line_with_raxml(self):
        return str(self.name) + "," + str(self.n_taxa) + "," + str(self.n_trees) + "," + str(self.n_reticulations) + "," + str(self.msa_size) + "," + str(self.sampling_type) + "," + str(self.simulation_type) + "," + str(self.likelihood_type) + "," + str(self.timeout) + "," + str(self.n_random_start_networks) + "," + str(self.n_parsimony_start_networks) + ",True"


class Result:
    def __init__(self, dataset):
        self.dataset = dataset
        self.bic_true = 0
        self.logl_true = 0
        self.bic_inferred = 0
        self.logl_inferred = 0
        self.bic_inferred_with_raxml = 0
        self.logl_inferred_with_raxml = 0
        self.bic_raxml = 0
        self.logl_raxml = 0
        self.n_reticulations_inferred = 0
        self.n_reticulations_inferred_with_raxml = 0
        self.topological_distances = {}
        self.topological_distances_with_raxml = {}
        self.rf_absolute_raxml = -1
        self.rf_relative_raxml = -1
        self.rf_absolute_inferred = -1
        self.rf_relative_inferred = -1
        self.rf_absolute_inferred_with_raxml = -1
        self.rf_relative_inferred_with_raxml = -1
        self.near_zero_branches_raxml = 0
        self.runtime_inference = 0
        self.runtime_inference_with_raxml = 0
        
    def get_csv_line(self):
        topo_scores_strings_ordered = [str(self.topological_distances[x]) for x in TOPOLOGICAL_DISTANCE_NAMES]
        return str(self.n_reticulations_inferred) + "," + str(self.bic_true) + "," + str(self.logl_true) + "," + str(self.bic_inferred) + "," + str(self.logl_inferred) + "," + str(self.bic_raxml) + "," + str(self.logl_raxml) + "," + str(self.rf_absolute_raxml) + "," + str(self.rf_relative_raxml) + "," + str(self.rf_absolute_inferred) + "," + str(self.rf_relative_inferred) + "," + str(self.near_zero_branches_raxml) + "," + str(self.runtime_inference) + "," + ",".join(topo_scores_strings_ordered)
        
    def get_csv_line_with_raxml(self):
        topo_scores_strings_ordered_with_raxml = [str(self.topological_distances_with_raxml[x]) for x in TOPOLOGICAL_DISTANCE_NAMES]
        return str(self.n_reticulations_inferred_with_raxml) + "," + str(self.bic_true) + "," + str(self.logl_true) + "," + str(self.bic_inferred_with_raxml) + "," + str(self.logl_inferred_with_raxml) + "," + str(self.bic_raxml) + "," + str(self.logl_raxml) + "," + str(self.rf_absolute_raxml) + "," + str(self.rf_relative_raxml) + "," + str(self.rf_absolute_inferred_with_raxml) + "," + str(self.rf_relative_inferred_with_raxml) + "," + str(self.near_zero_branches_raxml) + "," + str(self.runtime_inference_with_raxml) + "," + ",".join(topo_scores_strings_ordered_with_raxml)
        


from enum import Enum
    
    
class SamplingType(Enum):
    STANDARD = 1 # randomly choose which tree to sample, then sample equal number of sites for each sampled tree - this is the only mode that uses the n_trees or m parameter for sampling
    PERFECT_SAMPLING = 2 # sample each displayed tree, and as many site as expected by the tree probability
    PERFECT_UNIFORM_SAMPLING = 3 # sample each displayed tree, with the same number of sites per tree (ignoring reticulation probabilities)
    SINGLE_SITE_SAMPLING = 4 # sample each site individually, with the reticulation probabilities in mind
    
    
class SimulatorType(Enum):
    CELINE = 1 # use Celine's network topology simulator
    SARAH = 2 # use Sarah's ad-hoc network topology generator
    
    
class LikelihoodType(Enum):
    AVERAGE = 1 # use weighted average of displayed trees
    BEST = 2 # use best displayed tree


class BrlenLinkageType(Enum):
    SCALED = 1 # all partitions share the same brlens, each partition has its own scaling factor
    LINKED = 2 # all partitions share the same brlens. No sclaing factors.
    UNLINKED = 3 # each partitions has its own brlens


class StartType(Enum):
    RANDOM = 1 # start with random start trees and parsimony trees
    FROM_RAXML = 2 # start from the raxml-ng best tree
    ENDLESS = 3 # keep retrying with new random starting tree until timeout seconds have passed
    
    
class InferenceVariant:
    def __init__(self, likelihood_type, brlen_linkage_type, start_type, inferred_network_prefix):
        self.likelihood_type = likelihood_type
        self.brlen_linkage_type = brlen_linkage_type
        self.start_type = start_type
        self.inferred_network_path = inferred_network_prefix + "_" + str(self.likelihood_type) + "_" + str(self.brlen_linkage_type) + "_" + str(self.start_type) + "_inferred_network.nw"
        
        self.timeout = 0
        self.n_random_start_networks = 0
        self.n_parsimony_start_networks = 0
        self.runtime_inference = 0
        
        self.result = None
        
    def get_csv_line(self):
        return str(self.likelihood_type) + "," + str(self.brlen_linkage_type) + "," + str(self.start_type) + "," + str(self.timeout) + "," + str(self.n_random_start_networks) + "," + str(self.n_parsimony_start_networks) + "," + str(self.runtime_inference)

TOPOLOGICAL_DISTANCE_NAMES = ['hardwired_cluster_distance', 'softwired_cluster_distance', 'displayed_trees_distance', 'tripartition_distance', 'nested_labels_distance', 'path_multiplicity_distance']

DATASET_CSV_HEADER = "name,n_taxa,n_trees,n_reticulations,msa_size,sampling_type,simulation_type,celine_params,near_zero_branches_raxml"

INFERENCE_VARIANT_CSV_HEADER = "likelihood_type,brlen_linkage_type,start_type,timeout,n_random_start_networks,n_parsimony_start_networks,runtime_inference"

RESULT_CSV_HEADER = "n_reticulations_inferred,bic_true,logl_true,bic_inferred,logl_inferred,bic_raxml,logl_raxml,rf_absolute_raxml,rf_relative_raxml,rf_absolute_inferred,rf_relative_inferred" + "," + ",".join(TOPOLOGICAL_DISTANCE_NAMES)


class Dataset:
    def __init__(self):
        self.my_id = -1
        self.n_taxa = 0
        self.n_reticulations = 0
        self.n_trees = 0
        self.sites_per_tree = 0
        self.msa_path = ""
        self.partitions_path = "DNA"
        self.extracted_trees_path = ""
        self.true_network_path = ""
        self.raxml_tree_path = ""
        self.name = ""
        
        self.sampling_type = SamplingType.PERFECT_SAMPLING
        self.simulator_type = SimulatorType.CELINE
        self.celine_params = {}
        
        self.inference_variants = []
        self.near_zero_branches_raxml = -1

    def msa_size(self):
        return self.n_trees * self.sites_per_tree
        
        
    def get_csv_line(self):
        return str(self.name) + "," + str(self.n_taxa) + "," + str(self.n_trees) + "," + str(self.n_reticulations) + "," + str(self.msa_size) + "," + str(self.sampling_type) + "," + str(self.simulator_type) + "," + str(self.celine_params).replace(",", "|") + "," + str(self.near_zero_branches_raxml)


class Result:
    def __init__(self):
        self.bic_true = 0
        self.logl_true = 0
        self.bic_inferred = 0
        self.logl_inferred = 0
        self.bic_raxml = 0
        self.logl_raxml = 0
        self.n_reticulations_inferred = 0
        self.topological_distances = {}
        self.rf_absolute_raxml = -1
        self.rf_relative_raxml = -1
        self.rf_absolute_inferred = -1
        self.rf_relative_inferred = -1
        self.near_zero_branches_raxml = 0
        
    def get_csv_line(self):
        topo_scores_strings_ordered = [str(self.topological_distances[x]) for x in TOPOLOGICAL_DISTANCE_NAMES]
        return str(self.n_reticulations_inferred) + "," + str(self.bic_true) + "," + str(self.logl_true) + "," + str(self.bic_inferred) + "," + str(self.logl_inferred) + "," + str(self.bic_raxml) + "," + str(self.logl_raxml) + "," + str(self.rf_absolute_raxml) + "," + str(self.rf_relative_raxml) + "," + str(self.rf_absolute_inferred) + "," + str(self.rf_relative_inferred) + "," + ",".join(topo_scores_strings_ordered)
        


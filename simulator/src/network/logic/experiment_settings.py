from experiment_model import SamplingType, SimulatorType, LikelihoodType, BrlenLinkageType, StartType


class ExperimentSettings:
    def __init__(self):
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
        self.min_reticulation_prob = 0.1
        self.max_reticulation_prob = 0.9
        self.brlen_scalers = [1.0]


def small_tree():
    settings = ExperimentSettings()
    prefix = 'small_tree'
    settings.min_taxa = 4
    settings.max_taxa = 10
    settings.min_reticulations = 0
    settings.max_reticulations = 0
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.SCALED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [500, 1000]
    return (prefix, settings)


def small_network():
    settings = ExperimentSettings()
    prefix = 'small_network'
    settings.min_taxa = 4
    settings.max_taxa = 10
    settings.min_reticulations = 1
    settings.max_reticulations = 2
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [500, 1000]
    return (prefix, settings)


def small_network_single_debug():
    settings = ExperimentSettings()
    prefix = 'small_network_single_debug'
    settings.min_taxa = 4
    settings.max_taxa = 4
    settings.min_reticulations = 1
    settings.max_reticulations = 1
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    return (prefix, settings)


def larger_network():
    settings = ExperimentSettings()
    prefix = 'larger_network'
    settings.min_taxa = 20
    settings.max_taxa = 50
    settings.min_reticulations = 2
    settings.max_reticulations = -1
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [500, 1000]
    return (prefix, settings)

from experiment_model import SamplingType, SimulatorType, LikelihoodType, BrlenLinkageType, StartType


class ExperimentSettings:
    def __init__(self):
        self.min_taxa = 4
        self.max_taxa = 10
        self.min_reticulations = 1
        self.max_reticulations = 2
        self.min_reticulation_prob = 0.1
        self.max_reticulation_prob = 0.9

        self.use_fixed_simulation = False
        self.fixed_n_taxa = []
        self.fixed_n_reticulations = []
        self.fixed_reticulation_probs = [0.5]
        self.fixed_brlen_scalers = [1.0]

        self.sampling_types = [SamplingType.PERFECT_SAMPLING]
        self.simulator_types = [SimulatorType.CELINE]
        self.likelihood_types = [LikelihoodType.AVERAGE, LikelihoodType.BEST]
        self.brlen_linkage_types = [BrlenLinkageType.LINKED]
        self.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
        self.partition_sizes = [1000]
        self.use_partitioned_msa_types = [True]


def ten_taxa_change_reticulation_prob():
    settings = ExperimentSettings()
    prefix = 'ten_taxa_change_reticulation_prob'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [10]
    settings.fixed_n_reticulations = [1]
    settings.fixed_reticulation_probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def ten_taxa_change_brlen_scaler():
    settings = ExperimentSettings()
    prefix = 'ten_taxa_change_brlen_scaler'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [10]
    settings.fixed_n_reticulations = [1]
    settings.fixed_reticulation_probs = [0.5]
    settings.fixed_brlen_scalers = [1, 2, 4, 8]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def ten_taxa_change_reticulation_count():
    settings = ExperimentSettings()
    prefix = 'ten_taxa_change_reticulation_count'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [10]
    settings.fixed_n_reticulations = [1, 2, 3]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def ten_taxa_unpartitioned():
    settings = ExperimentSettings()
    prefix = 'ten_taxa_unpartitioned'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [10]
    settings.fixed_n_reticulations = [1]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    settings.use_partitioned_msa_types = [True, False]
    return (prefix, settings)


def gather_labeled_settings():
    setups = {}
    setups['ten_taxa_change_reticulation_prob'] = ten_taxa_change_reticulation_prob()
    setups['ten_taxa_change_brlen_scaler'] = ten_taxa_change_brlen_scaler()
    setups['ten_taxa_change_reticulation_count'] = ten_taxa_change_reticulation_count()
    setups['ten_taxa_unpartitioned'] = ten_taxa_unpartitioned()
    return setups
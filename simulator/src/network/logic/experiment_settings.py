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


def exp_change_reticulation_prob(n_taxa):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_change_reticulation_prob'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [1]
    settings.fixed_reticulation_probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_change_brlen_scaler(n_taxa):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_change_brlen_scaler'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [1]
    settings.fixed_reticulation_probs = [0.5]
    settings.fixed_brlen_scalers = [1, 2, 4, 8]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_change_reticulation_count(n_taxa):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_change_reticulation_count'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [1, 2, 3]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_unpartitioned(n_taxa):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_unpartitioned'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [1]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    settings.use_partitioned_msa_types = [True, False]
    return (prefix, settings)


def smoke_test_fixed():
    settings = ExperimentSettings()
    prefix = 'smoke_test_fixed'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.AVERAGE]
    settings.partition_sizes = [100]
    settings.fixed_n_taxa = [4]
    settings.fixed_n_reticulations = [0]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    settings.use_partitioned_msa_types = [True]
    return (prefix, settings)


def gather_labeled_settings():
    setups = {}
    setups['t_15_change_reticulation_prob'] = exp_change_reticulation_prob(15)
    setups['t_15_change_brlen_scaler'] = exp_change_brlen_scaler(15)
    setups['t_15_change_reticulation_count'] = exp_change_reticulation_count(15)
    setups['t_15_unpartitioned'] = exp_unpartitioned(15)
    setups['smoke_test_fixed'] = smoke_test_fixed()
    return setups
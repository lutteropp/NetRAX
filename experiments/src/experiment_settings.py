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


def exp_change_reticulation_prob(n_taxa, n_reticulations, with_random=False):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_change_reticulation_prob'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    if with_random:
        settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    else:
        settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [n_reticulations]
    settings.fixed_reticulation_probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_change_reticulation_prob_test(n_taxa, n_reticulations, with_random=False):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_change_reticulation_prob_test'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    if with_random:
        settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    else:
        settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [100]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [n_reticulations]
    settings.fixed_reticulation_probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_change_brlen_scaler(n_taxa, n_reticulations, with_random=False):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_change_brlen_scaler'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    if with_random:
        settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    else:
        settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [n_reticulations]
    settings.fixed_reticulation_probs = [0.5]
    settings.fixed_brlen_scalers = [1, 2, 4, 8]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_change_reticulation_count(n_taxa, with_random=False):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_change_reticulation_count'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    if with_random:
        settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    else:
        settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [1, 2, 3, 4]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_standard(n_taxa, n_reticulations, with_random=False):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_standard'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    if with_random:
        settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    else:
        settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [n_reticulations]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_partition_size(n_taxa, n_reticulations, with_random=False):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_partition_size'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    if with_random:
        settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    else:
        settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [100, 500, 1000, 5000, 10000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [n_reticulations]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_brlen_linkage(n_taxa, n_reticulations, with_random=False):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_change_reticulation_count'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    if with_random:
        settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    else:
        settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED, BrlenLinkageType.UNLINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [n_reticulations]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_standard_random(n_taxa, n_reticulations, with_random=False):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_standard_random'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    if with_random:
        settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    else:
        settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST, LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [n_reticulations]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    return (prefix, settings)


def exp_unpartitioned(n_taxa, n_reticulations, with_random=False):
    settings = ExperimentSettings()
    prefix = 't_' + str(n_taxa) + '_unpartitioned'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    if with_random:
        settings.start_types = [StartType.FROM_RAXML, StartType.RANDOM]
    else:
        settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.AVERAGE]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [n_reticulations]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    settings.use_partitioned_msa_types = [True, False]
    return (prefix, settings)


def smoke_test_fixed(n_taxa, n_reticulations):
    settings = ExperimentSettings()
    prefix = 'smoke_test_fixed'
    settings.sampling_types = [SamplingType.PERFECT_SAMPLING]
    settings.start_types = [StartType.FROM_RAXML]
    settings.brlen_linkage_types = [BrlenLinkageType.LINKED]
    settings.likelihood_types = [LikelihoodType.BEST]
    settings.partition_sizes = [1000]
    settings.fixed_n_taxa = [n_taxa]
    settings.fixed_n_reticulations = [n_reticulations]
    settings.fixed_reticulation_probs = [0.5]
    settings.use_fixed_simulation = True
    settings.use_partitioned_msa_types = [True]
    return (prefix, settings)


def gather_labeled_settings():
    setups={}
    n_taxa_list=[10,15,20,25,30,35,40]
    n_reticulations_list=[1,2,3,4]

    for n_taxa in n_taxa_list:
        for n_reticulations in n_reticulations_list:
            setups['t_'+str(n_taxa)+'_r_'+str(n_reticulations)+'_change_reticulation_prob'] = exp_change_reticulation_prob(n_taxa, n_reticulations)
            setups['t_'+str(n_taxa)+'_r_'+str(n_reticulations)+'_change_brlen_scaler'] = exp_change_brlen_scaler(n_taxa, n_reticulations)
            setups['t_'+str(n_taxa)+'_r_'+str(n_reticulations)+'_standard'] = exp_standard(n_taxa, n_reticulations)
            setups['t_'+str(n_taxa)+'_r_'+str(n_reticulations)+'_standard_multi'] = exp_standard(n_taxa, n_reticulations)
            setups['t_'+str(n_taxa)+'_r_'+str(n_reticulations)+'_unpartitioned'] = exp_unpartitioned(n_taxa, n_reticulations)
            setups['t_'+str(n_taxa)+'_r_'+str(n_reticulations)+'_partition_size'] = exp_partition_size(n_taxa, n_reticulations)

            setups['t_'+str(n_taxa)+'_r_'+str(n_reticulations)+'_standard_random'] = exp_standard(n_taxa, n_reticulations, True)
            setups['t_'+str(n_taxa)+'_r_'+str(n_reticulations)+'_brlen_linkage'] = exp_brlen_linkage(n_taxa, n_reticulations)

    setups['smoke_test_fixed'] = smoke_test_fixed(40, 4)
    setups['smoke_test_medium'] = smoke_test_fixed(20, 2)
    setups['smoke_test_small'] = smoke_test_fixed(10, 1)
    setups['smoke_test_tiny'] = smoke_test_fixed(4, 1)
    return setups

import os
from dataset_builder import build_dataset
from evaluate_experiments import SamplingType, SimulationType, LikelihoodType, run_inference_and_evaluate, write_results_to_csv

WANTED_NUM_TAXA = [5, 10, 15, 50, 100]
WANTED_NUM_RETICULATIONS = [0, 1, 2, 3, 4, 5]
WANTED_MSA_SIZES = [1000,5000,10000] # wanted number of sites in the total msa
WANTED_M_VALUES = [1, 5, 10, 20, 30] # wanted values for m. Used to generate m*2^k trees with k being the number of reticulations in the network.
WANTED_SIMULATORS = [SimulationType.SARAH] #[SimulationType.CELINE]
WANTED_SAMPLINGS = [SamplingType.STANDARD, SamplingType.PERFECT_SAMPLING, SamplingType.PERFECT_UNIFORM_SAMPLING, SamplingType.SINGLE_SITE_SAMPLING]
WANTED_LIKELIHOODS = [LikelihoodType.AVERAGE, LikelihoodType.BEST]

WANTED_TIMEOUTS = [0, 60] # Timeout of zero means a single netrax random starting network, timeout larger than zero means continue starting from other random networks until timeout has been reached


def build_dataset(n_taxa, n_reticulations, msa_size, m, simulator, sampling, likelihood, timeout):
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    name = "datasets/" + str(simulator) + "_" + str(sampling) + "_" + str(likelihood) + "_" + str(n_taxa) + "_" + str(n_reticulations) + "_" + str(msa_size) + "_" + str(m) + "_" + str(timeout) 
    return build_dataset(n_taxa, n_reticulations, msa_size, sampling, likelihood, name, timeout, m)
    

def run_experiments(wanted_num_taxa, wanted_num_reticulations, wanted_msa_sizes, wanted_m_values, wanted_simulators, wanted_samplings, wanted_timeouts):
    datasets = []
    results = []
    for simulator in wanted_simulators:
        for sampling in wanted_samplings:
            for m in wanted_m_values:
                if sampling != SamplingType.STANDARD and m != wanted_m_values[0]:
                    continue
                for n_taxa in wanted_num_taxa:
                    for n_reticulations in wanted_num_reticulations:
                        for msa_size in wanted_msa_sizes:
                            for likelihood in wanted_likelihoods:
                                for timeout in wanted_timeouts:
                                    my_dataset = build_dataset(n_taxa, n_reticulations, msa_size, m, simulator, sampling, likelihood, timeout)
                                    result = run_inference_and_evaluate([my_dataset])
                                    datasets.append(my_dataset)
                                    results.append(result)
    write_results_to_csv(results, "all_results.csv")

    
if __name__ == "__main__":
    run_experiments(WANTED_NUM_TAXA, WANTED_NUM_RETICULATIONS, WANTED_MSA_SIZES, WANTED_M_VALUES, WANTED_SIMULATORS, WANTED_SAMPLINGS, WANTED_LIKELIHOODS, WANTED_TIMEOUTS)


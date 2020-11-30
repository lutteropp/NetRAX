import matplotlib
import pandas as pd


def create_plots_internal(prefix, data, simulator_type, sampling_type, likelihood_type):
    pass


def create_plots(prefix):
    data = pd.read_csv(prefix + "_results.csv")
    msa_sizes = data.msa_size.unique()
    simulator_types = data.simulation_type.unique()
    sampling_types = data.sampling_type.unique()
    likelihood_types = data.likelihood_type.unique()
    for simulator_type in simulator_types:
        for sampling_type in sampling_types:
            for msa_size in msa_sizes:
                for likelihood_type in likelihood_types:
                    create_plots_internal(prefix, data, simulator_type, sampling_type, likelihood_type)

if __name__ == "__main__":
    create_plots("small_tree")

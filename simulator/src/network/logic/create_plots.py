import matplotlib
import pandas as pd

def create_plots(prefix):
    data = pd.read_csv(prefix + "_results.csv")
    msa_sizes = data.msa_size.unique()
    simulator_types = data.simulation_type.unique()
    sampling_types = data.sampling_type.unique()
    likelihood_types = data.likelihood_type.unique()
    print(msa_sizes)

if __name__ == "__main__":
    create_plots("small_tree")

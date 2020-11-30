import matplotlib
import pandas as pd
import os

def merge_multi(dataframes, column_name):
    merged = dataframes[0]
    for i in range(1, len(dataframes)):
        merged = pd.merge(merged, dataframes[i], on=column_name)
    return merged


def create_plots_internal(prefix, data, simulator_type, sampling_type, msa_size, likelihood_type):
    #pd.set_option('display.max_columns', None)
    name_prefix = str(simulator_type) + "_simulator_" + str(sampling_type) + "_sampling_" + str(msa_size) + "_msasize_" + str(likelihood_type) + "_likelihood"
    filtered_data = data.loc[(data['simulation_type'] == simulator_type) & (data['sampling_type'] == sampling_type) & (data['msa_size'] == msa_size) & (data['likelihood_type'] == likelihood_type)]
    # BIC plot
    true_simulated_network_bics = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'bic_true']]
    inferred_network_bics = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'bic_inferred']]
    inferred_network_with_raxml_bics = filtered_data.loc[filtered_data['start_from_raxml'] == True][['name', 'bic_inferred']].rename(columns={'bic_inferred': 'bic_inferred_with_raxml'})
    raxml_bics = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'bic_raxml']]
    merged_df = merge_multi([true_simulated_network_bics, inferred_network_bics, inferred_network_with_raxml_bics, raxml_bics], 'name')
    print(merged_df.head())
    # Logl plot
    # relative RF-distance plot
    # num near-zero raxml-ng branches plot
    pass


def create_plots(prefix):
    if not os.path.exists('plots_' + prefix):
        os.makedirs('plots_' + prefix)
    data = pd.read_csv(prefix + "_results.csv", sep=',|;', index_col=False, engine='python')
    msa_sizes = data.msa_size.unique()
    simulator_types = data.simulation_type.unique()
    sampling_types = data.sampling_type.unique()
    likelihood_types = data.likelihood_type.unique()
    for simulator_type in simulator_types:
        for sampling_type in sampling_types:
            for msa_size in msa_sizes:
                for likelihood_type in likelihood_types:
                    create_plots_internal(prefix, data, simulator_type, sampling_type, msa_size, likelihood_type)


if __name__ == "__main__":
    create_plots("small_tree")

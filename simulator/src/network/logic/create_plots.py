import matplotlib.pyplot as plt
import pandas as pd
import os

def merge_multi(dataframes, column_name):
    merged = dataframes[0]
    for i in range(1, len(dataframes)):
        merged = pd.merge(merged, dataframes[i], on=column_name)
    return merged


def create_bic_plot(prefix, name_prefix, filtered_data):
    filepath = 'plots_' + prefix + '/' + name_prefix + '_bic_plot.png'

    true_simulated_network_bics = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'bic_true']]
    inferred_network_bics = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'bic_inferred']]
    inferred_network_with_raxml_bics = filtered_data.loc[filtered_data['start_from_raxml'] == True][['name', 'bic_inferred']].rename(columns={'bic_inferred': 'bic_inferred_with_raxml'})
    raxml_bics = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'bic_raxml']]
    merged_df = merge_multi([true_simulated_network_bics, inferred_network_bics, inferred_network_with_raxml_bics, raxml_bics], 'name')
    
    bic_dict_list = []
    for _, row in merged_df.iterrows():
        act_entry = {}
        act_entry['id'] = int(row['name'].split('/')[1].split('_')[0])
        act_entry['rel_diff_bic_inferred'] = float(row['bic_inferred'] - row['bic_true']) / row['bic_true']
        act_entry['rel_diff_bic_inferred_with_raxml'] = float(row['bic_inferred_with_raxml'] - row['bic_true']) / row['bic_true']
        act_entry['rel_diff_bic_raxml'] = float(row['bic_raxml'] - row['bic_true']) / row['bic_true']
        bic_dict_list.append(act_entry)
    df_bic = pd.DataFrame(bic_dict_list)
    
    df_bic.plot(x="id", y=["rel_diff_bic_inferred", "rel_diff_bic_inferred_with_raxml", "rel_diff_bic_raxml"])
    plt.title("Relative BIC difference for " + name_prefix.replace('_',' '), wrap=True)
    plt.savefig(filepath)
    #plt.show()


def create_plots_internal(prefix, data, simulator_type, sampling_type, msa_size, likelihood_type):
    name_prefix = str(simulator_type) + "_" + str(sampling_type) +'_' + str(msa_size) + "_msasize_" + str(likelihood_type)
    filtered_data = data.loc[(data['simulation_type'] == simulator_type) & (data['sampling_type'] == sampling_type) & (data['msa_size'] == msa_size) & (data['likelihood_type'] == likelihood_type)]
    # BIC plot
    create_bic_plot(prefix, name_prefix, filtered_data)
    
    
    # Logl plot
    # relative RF-distance plot
    # num near-zero raxml-ng branches plot
    pass


def create_plots(prefix):
    pd.set_option('display.max_columns', None)
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

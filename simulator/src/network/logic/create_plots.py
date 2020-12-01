import matplotlib.pyplot as plt
import pandas as pd
import os
import collections
#import seaborn as sns
#import numpy as np

def merge_multi(dataframes, column_name):
    merged = dataframes[0]
    for i in range(1, len(dataframes)):
        merged = pd.merge(merged, dataframes[i], on=column_name)
    return merged


def create_bic_plot(prefix, name_prefix, filtered_data):
    plot_filepath = 'plots_' + prefix + '/' + name_prefix + '_bic_plot.png'
    stats_filepath = 'plots_' + prefix + '/' + name_prefix + '_bic_stats.png'

    true_simulated_network_bics = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'bic_true']]
    inferred_network_bics = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'bic_inferred']]
    inferred_network_with_raxml_bics = filtered_data.loc[filtered_data['start_from_raxml'] == True][['name', 'bic_inferred']].rename(columns={'bic_inferred': 'bic_inferred_with_raxml'})
    raxml_bics = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'bic_raxml']]
    merged_df = merge_multi([true_simulated_network_bics, inferred_network_bics, inferred_network_with_raxml_bics, raxml_bics], 'name')
    
    counts = collections.defaultdict(int)
    bic_dict_list = []
    for _, row in merged_df.iterrows():
        act_entry = {}
        act_entry['id'] = int(row['name'].split('/')[1].split('_')[0])
        act_entry['rel_diff_bic_inferred'] = float(row['bic_inferred'] - row['bic_true']) / row['bic_true']
        act_entry['rel_diff_bic_inferred_with_raxml'] = float(row['bic_inferred_with_raxml'] - row['bic_true']) / row['bic_true']
        act_entry['rel_diff_bic_raxml'] = float(row['bic_raxml'] - row['bic_true']) / row['bic_true']
        bic_dict_list.append(act_entry)
        
        if row['bic_inferred'] <= row['bic_true']:
            counts['bic_inferred_better_or_equal_than_true'] += 1
        else:
            counts['bic_inferred_worse_than_true'] += 1
        if row['bic_inferred_with_raxml'] <= row['bic_true']:
            counts['bic_inferred_with_raxml_better_or_equal_than_true'] += 1
        else:
            counts['bic_inferred_with_raxml_worse_than_true'] += 1
        if row['bic_raxml'] <= row['bic_true']:
            counts['bic_raxml_better_or_equal_than_true'] += 1
        else:
            counts['bic_raxml_worse_than_true'] += 1
        
    plt.tight_layout()
    df_bic = pd.DataFrame(bic_dict_list)
    df_bic.plot(x="id", y=["rel_diff_bic_inferred", "rel_diff_bic_inferred_with_raxml", "rel_diff_bic_raxml"])
    plt.title(prefix + "\nRelative BIC difference for\n" + name_prefix.replace('_',' ') + "\nNegative value means BIC got better!", wrap=True, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    
    df_bic_stats = pd.DataFrame([counts])
    df_bic_stats.plot(kind='bar')
    plt.title(prefix + "\nRelative BIC statistics for\n" + name_prefix.replace('_',' '), wrap=True, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(stats_filepath, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    
    
def create_logl_plot(prefix, name_prefix, filtered_data):
    plot_filepath = 'plots_' + prefix + '/' + name_prefix + '_logl_plot.png'
    stats_filepath = 'plots_' + prefix + '/' + name_prefix + '_logl_stats.png'

    true_simulated_network_logls = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'logl_true']]
    inferred_network_logls = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'logl_inferred']]
    inferred_network_with_raxml_logls = filtered_data.loc[filtered_data['start_from_raxml'] == True][['name', 'logl_inferred']].rename(columns={'logl_inferred': 'logl_inferred_with_raxml'})
    raxml_logls = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'logl_raxml']]
    merged_df = merge_multi([true_simulated_network_logls, inferred_network_logls, inferred_network_with_raxml_logls, raxml_logls], 'name')
    
    counts = collections.defaultdict(int)
    logl_dict_list = []
    for _, row in merged_df.iterrows():
        act_entry = {}
        act_entry['id'] = int(row['name'].split('/')[1].split('_')[0])
        act_entry['rel_diff_logl_inferred'] = float(row['logl_inferred'] - row['logl_true']) / row['logl_true']
        act_entry['rel_diff_logl_inferred_with_raxml'] = float(row['logl_inferred_with_raxml'] - row['logl_true']) / row['logl_true']
        act_entry['rel_diff_logl_raxml'] = float(row['logl_raxml'] - row['logl_true']) / row['logl_true']
        logl_dict_list.append(act_entry)
        
        if row['logl_inferred'] <= row['logl_true']:
            counts['logl_inferred_better_or_equal_than_true'] += 1
        else:
            counts['logl_inferred_worse_than_true'] += 1
        if row['logl_inferred_with_raxml'] <= row['logl_true']:
            counts['logl_inferred_with_raxml_better_or_equal_than_true'] += 1
        else:
            counts['logl_inferred_with_raxml_worse_than_true'] += 1
        if row['logl_raxml'] <= row['logl_true']:
            counts['logl_raxml_better_or_equal_than_true'] += 1
        else:
            counts['logl_raxml_worse_than_true'] += 1
        
    plt.tight_layout()
    df_logl = pd.DataFrame(logl_dict_list)
    df_logl.plot(x="id", y=["rel_diff_logl_inferred", "rel_diff_logl_inferred_with_raxml", "rel_diff_logl_raxml"])
    plt.title(prefix + "\nRelative logl difference for\n" + name_prefix.replace('_',' ') + "\Positive value means logl got better!", wrap=True, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    
    df_logl_stats = pd.DataFrame([counts])
    df_logl_stats.plot(kind='bar')
    plt.title(prefix + "\nRelative logl statistics for\n" + name_prefix.replace('_',' '), wrap=True, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(stats_filepath, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    
    
def create_relative_rf_dist_plot(prefix, name_prefix, filtered_data):
    plot_filepath = 'plots_' + prefix + '/' + name_prefix + '_rfdist_plot.png'
    stats_filepath = 'plots_' + prefix + '/' + name_prefix + '_rfdist_stats.png'

    inferred_network_rfdist = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'rf_relative_inferred']]
    inferred_network_with_raxml_rfdist = filtered_data.loc[filtered_data['start_from_raxml'] == True][['name', 'rf_relative_inferred']].rename(columns={'rf_relative_inferred': 'rf_relative_inferred_with_raxml'})
    raxml_rfdist = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'rf_relative_raxml']]
    merged_df = merge_multi([inferred_network_rfdist, inferred_network_with_raxml_rfdist, raxml_rfdist], 'name')
    
    counts = collections.defaultdict(int)
    logl_dict_list = []
    for _, row in merged_df.iterrows():
        act_entry = {}
        act_entry['id'] = int(row['name'].split('/')[1].split('_')[0])
        act_entry['rf_relative_inferred'] = row['rf_relative_inferred']
        act_entry['rf_relative_inferred_with_raxml'] = row['rf_relative_inferred_with_raxml']
        act_entry['rf_relative_raxml'] = row['rf_relative_raxml']
        logl_dict_list.append(act_entry)
        
        if row['rf_relative_inferred'] > 0:
            counts['rfdist_inferred_greater_zero'] += 1
        else:
            counts['rfdist_inferred_zero'] += 1
        if row['rf_relative_inferred_with_raxml'] > 0:
            counts['rfdist_inferred_with_raxml_greater_zero'] += 1
        else:
            counts['rfdist_inferred_with_raxml_zero'] += 1
        if row['rf_relative_raxml'] > 0:
            counts['rfdist_raxml_greater_zero'] += 1
        else:
            counts['rfdist_raxml_zero'] += 1
        
    plt.tight_layout()
    df_logl = pd.DataFrame(logl_dict_list)
    df_logl.plot(x="id", y=["rf_relative_inferred", "rf_relative_inferred_with_raxml", "rf_relative_raxml"])
    plt.title(prefix + "\nRelative RF-distance to simulated tree for\n" + name_prefix.replace('_',' '), wrap=True, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    
    df_logl_stats = pd.DataFrame([counts])
    df_logl_stats.plot(kind='bar')
    plt.title(prefix + "\nRelative RF-distance to simulated tree statistics for\n" + name_prefix.replace('_',' '), wrap=True, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(stats_filepath, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    
    
def create_num_nearzero_raxml_branches_plot(prefix, name_prefix, filtered_data):
    hist_filepath = 'plots_' + prefix + '/' + name_prefix + '_raxml_nonzero_branches_histogram.png'
    raxml_nonzero_branches = filtered_data.loc[filtered_data['start_from_raxml'] == False][['name', 'near_zero_branches_raxml']]
    
    counts = collections.defaultdict(int)
    for _, row in raxml_nonzero_branches.iterrows():
        counts[row['near_zero_branches_raxml']] += 1
    
    plt.bar(list(counts.keys()), counts.values())
    plt.tight_layout()
    plt.title(prefix + "\nNumber of near-zero branches in raxml-ng tree for\n" + name_prefix.replace('_',' '), wrap=True, fontsize=8)
    plt.savefig(hist_filepath, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    

def create_plots_internal(prefix, data, simulator_type, sampling_type, msa_size, likelihood_type):
    name_prefix = str(simulator_type) + "_" + str(sampling_type) +'_' + str(msa_size) + "_msasize_" + str(likelihood_type)
    filtered_data = data.loc[(data['simulation_type'] == simulator_type) & (data['sampling_type'] == sampling_type) & (data['msa_size'].isin([msa_size + x for x in range(50)])) & (data['likelihood_type'] == likelihood_type)]
    if not filtered_data.empty:
        # Create the plots for BIC, logl, rf-dist, nearzero-branches
        create_bic_plot(prefix, name_prefix, filtered_data)
        create_logl_plot(prefix, name_prefix, filtered_data)
        create_relative_rf_dist_plot(prefix, name_prefix, filtered_data)
        create_num_nearzero_raxml_branches_plot(prefix, name_prefix, filtered_data)


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
    #create_plots("small_network")

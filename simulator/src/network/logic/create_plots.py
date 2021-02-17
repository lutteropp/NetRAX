import argparse
from prettytable import PrettyTable

#%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


def import_dataframe(prefix):
    df = pd.read_csv(prefix + "_results.csv")
    df['bic_diff'] = (df['bic_true'] - df['bic_inferred'])
    df['logl_diff'] = (df['logl_true'] - df['logl_inferred'])
    df['bic_diff_relative'] = (df['bic_true'] - df['bic_inferred']) / df['bic_true']
    df['logl_diff_relative'] = (df['logl_true'] - df['logl_inferred']) / df['logl_true']
    df['msa_patterns_relative'] = df['msa_patterns'] / df['msa_size']
    return df


def generate_ascii_table(df):
    x = PrettyTable()
    x.field_names = df.columns.tolist()
    for row in df.values:
        x.add_row(row)
    print(x)
    return x


def report_bic_better_or_equal(df):
    cnt = len(df[df['bic_inferred'] <= df['bic_true']])
    return str(cnt) + " (" + "%.2f" % (float(cnt*100)/len(df)) + " %)"


def report_bic_worse(df):
    cnt = len(df[df['bic_inferred'] > df['bic_true']])
    return str(cnt) + " (" + "%.2f" % (float(cnt*100)/len(df)) + " %)"


def report_logl_better_or_equal(df):
    cnt = len(df[df['logl_inferred'] >= df['logl_true']])
    return str(cnt) + " (" + "%.2f" % (float(cnt*100)/len(df)) + " %)"


def report_logl_worse(df):
    cnt = len(df[df['logl_inferred'] < df['logl_true']])
    return str(cnt) + " (" + "%.2f" % (float(cnt*100)/len(df)) + " %)"


def report_reticulations_less(df):
    cnt = len(df[df['n_reticulations_inferred'] < df['n_reticulations']])
    return str(cnt) + " (" + "%.2f" % (float(cnt*100)/len(df)) + " %)"


def report_reticulations_equal(df):
    cnt = len(df[df['n_reticulations_inferred'] == df['n_reticulations']])
    return str(cnt) + " (" + "%.2f" % (float(cnt*100)/len(df)) + " %)"


def report_reticulations_more(df):
    cnt = len(df[df['n_reticulations_inferred'] > df['n_reticulations']])
    return str(cnt) + " (" + "%.2f" % (float(cnt*100)/len(df)) + " %)"


def plot_relative_quality_stats(df):
    plt.figure()
    fig, axes = plt.subplots(2, 2)
    fig.suptitle("Relative Quality Statistics")
    df['bic_diff_relative'].plot.hist(bins=100, alpha=0.5, title='Relative BIC difference (>0 means better)\n (bic_true - bic_inferred) / bic_true', ax=axes[0][0])
    df['logl_diff_relative'].plot.hist(bins=100, alpha=0.5, title='Relative loglh difference (<0 means better)\n (logl_true - logl_inferred) / logl_true', ax=axes[0][1])
    df['bic_diff'].plot.hist(bins=100, alpha=0.5, title='Absolute BIC difference (>0 means better)\n (bic_true - bic_inferred)', ax=axes[1][0])
    df['logl_diff'].plot.hist(bins=100, alpha=0.5, title='Absolute loglh difference (<0 means better)\n (logl_true - logl_inferred)', ax=axes[1][1])
    plt.tight_layout()
    plt.show()


def quality_stats(df):
    df_likelihood_average = df.query('likelihood_type == "AVERAGE"')
    df_likelihood_best = df.query('likelihood_type == "BEST"')
    data = [['Inferred BIC better or equal',report_bic_better_or_equal(df_likelihood_average),report_bic_better_or_equal(df_likelihood_best),report_bic_better_or_equal(df)],
            ['Inferred BIC worse',report_bic_worse(df_likelihood_average),report_bic_worse(df_likelihood_best),report_bic_worse(df)],
            ['Inferred logl better or equal',report_logl_better_or_equal(df_likelihood_average),report_logl_better_or_equal(df_likelihood_best),report_logl_better_or_equal(df)],
            ['Inferred logl worse', report_logl_worse(df_likelihood_average), report_logl_worse(df_likelihood_best),report_logl_worse(df)],
            ['Inferred n_reticulations less', report_reticulations_less(df_likelihood_average),report_reticulations_less(df_likelihood_best),report_reticulations_less(df)],
            ['Inferred n_reticulations equal', report_reticulations_equal(df_likelihood_average),report_reticulations_equal(df_likelihood_best),report_reticulations_equal(df)],
            ['Inferred n_reticulations more', report_reticulations_more(df_likelihood_average),report_reticulations_more(df_likelihood_best),report_reticulations_more(df)]]
    data_df = pd.DataFrame(data, columns=['', 'LikelihoodType.AVERAGE', 'LikelihoodType.BEST', 'Overall'])
    generate_ascii_table(data_df)
    plot_relative_quality_stats(df)
    
    
def plot_weirdness_stats(df):
    plt.figure()
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Network Weirdness Statistics")
    df['true_network_weirdness'].plot.hist(bins=10, alpha=0.5, range=(0,1), title='True network weirdness', ax=axes[0])
    df['near_zero_branches_raxml'].plot.hist(bins=10, alpha=0.5, title='Near-zero branches raxml', ax=axes[1])
    plt.tight_layout()
    plt.show()
    

def distances_dendroscope(df):
    plt.figure()
    fig, axes = plt.subplots(3, 2, constrained_layout=True)
    fig.suptitle("Topological Network Distances")
    df['hardwired_cluster_distance'].plot.hist(bins=10, alpha=0.5, title='Hardwired cluster distance', ax=axes[0,0])
    df['softwired_cluster_distance'].plot.hist(bins=10, alpha=0.5, title='Softwired cluster distance', ax=axes[0,1])
    df['displayed_trees_distance'].plot.hist(bins=10, alpha=0.5, title='Displayed trees distance', ax=axes[1,0])
    df['tripartition_distance'].plot.hist(bins=10, alpha=0.5, title='Tripartition distance', ax=axes[1,1])
    df['nested_labels_distance'].plot.hist(bins=10, alpha=0.5, title='Nested labels distance', ax=axes[2,0])
    df['path_multiplicity_distance'].plot.hist(bins=10, alpha=0.5, title='Path multiplicity distance', ax=axes[2,1])
    plt.show()

    if 'unrooted_softwired_distance' in df:
        plt.figure()
        df['unrooted_softwired_distance'].plot.hist(bins=100, alpha=0.5, range=(0,1), title='Unrooted softwired distance')
        plt.show()


def distances_netrax(df):
    plt.figure()
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle("Unrooted Topological Network Distances")
    df['unrooted_softwired_network_distance'].plot.hist(bins=10, alpha=0.5, title='Unrooted softwired cluster distance', ax=axes[0])
    df['unrooted_hardwired_network_distance'].plot.hist(bins=10, alpha=0.5, title='Unrooted hardwired cluster distance', ax=axes[1])
    df['unrooted_displayed_trees_distance'].plot.hist(bins=10, alpha=0.5, title='Unrooted displayed trees distance', ax=axes[2])
    plt.show()

    plt.figure()
    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    fig.suptitle("Rooted Topological Network Distances")
    df['rooted_softwired_network_distance'].plot.hist(bins=10, alpha=0.5, title='Rooted softwired cluster distance', ax=axes[0,0])
    df['rooted_hardwired_network_distance'].plot.hist(bins=10, alpha=0.5, title='Rooted hardwired cluster distance', ax=axes[0,1])
    df['rooted_displayed_trees_distance'].plot.hist(bins=10, alpha=0.5, title='Rooted displayed trees distance', ax=axes[0,2])

    df['rooted_tripartition_distance'].plot.hist(bins=10, alpha=0.5, title='Rooted tripartition distance', ax=axes[1,0])
    df['rooted_path_multiplicity_distance'].plot.hist(bins=10, alpha=0.5, title='Rooted path multiplicity distance', ax=axes[1,1])
    df['rooted_nested_labels_distance'].plot.hist(bins=10, alpha=0.5, title='Rooted nested labels distance', ax=axes[1,2])
    plt.show()

    
def plots_setup():
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')


def plot_dataset_size(df):
    print("Total number of datasets: " + str(len(df)))
    plt.figure()
    fig, axes = plt.subplots(2, 3)
    fig.suptitle("Simulated Dataset Stats")
    df['n_taxa'].plot.hist(bins=10, alpha=0.5, title='Number of taxa', ax=axes[0][0])
    df['n_reticulations'].plot.hist(bins=10, alpha=0.5, title='Number of reticulations', ax=axes[0][1])
    df['msa_size'].plot.hist(bins=10, alpha=0.5, title='MSA size', ax=axes[0][2])
    df['msa_patterns'].plot.hist(bins=10, alpha=0.5, title='MSA patterns (absolute count)', ax=axes[1][0])
    df['msa_patterns_relative'].plot.hist(bins=10, alpha=0.5, range=(0,1), title='MSA patterns (fraction)', ax=axes[1][1])
    plt.tight_layout()
    plt.show()


def show_stats(df):
    plot_dataset_size(df)
    plot_weirdness_stats(df)
    if 'brlen_scaler' in df:
        show_brlen_scaler_effects(df)


def show_pattern_quality_effects(df):
    print('correlation between number of MSA patterns and better-or-equal inferred BIC')

    pattern_sizes = sorted(list(df['msa_patterns'].unique()))
    pattern_sizes_yvals = []

    for psize in pattern_sizes:
        df_patterns = df.query('msa_patterns == ' + str(psize))
        percentage_good = float(len(df_patterns[df_patterns['bic_inferred'] <= df_patterns['bic_true']])) / len(df_patterns)
        pattern_sizes_yvals.append(percentage_good)
    plt.scatter(pattern_sizes, pattern_sizes_yvals)
    plt.show()

    print(df['msa_patterns'].corr(df['bic_diff']))


def show_brlen_scaler_effects(df):
    scalers = df.brlen_scaler.unique()
    plt.figure()
    fig, axes = plt.subplots(2, len(scalers))
    fig.suptitle("Effects of brlen_scaler choice on number of MSA patterns")
    for i in range(len(scalers)):
        scaler = scalers[i]
        df_scaled = df.query('brlen_scaler == ' + str(scaler))
        df_scaled['msa_patterns_relative'].plot.hist(bins=10, alpha=0.5, range=(0,1), title='MSA patterns (fraction) for brlen_scaler ' + str(scaler), ax=axes[0][i])
        df_scaled['msa_patterns'].plot.hist(bins=10, alpha=0.5, title='MSA patterns (absolute count) for brlen_scaler ' + str(scaler), ax=axes[1][i])
    plt.show()


def show_plots(df):
    quality_stats(df)
    print("")
    #distances_dendroscope(df)
    distances_netrax(df)
    #show_pattern_quality_effects(df)


def parse_command_line_arguments_plots():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="small_network")
    args = CLI.parse_args()
    return args.prefix


if __name__ == "__main__":
    prefix = parse_command_line_arguments_plots()
    df = import_dataframe(prefix)
    show_stats(df)
    show_plots(df)

import argparse

#%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


def import_dataframe(prefix):
    df = pd.read_csv(prefix + "_results.csv")
    df['bic_diff'] = (df['bic_true'] - df['bic_inferred']) / df['bic_true']
    df['logl_diff'] = (df['logl_true'] - df['logl_inferred']) / df['logl_true']
    return df


def cm_to_inch(value):
    return value/2.54


def bic_logl_stats(df):
    bic_better_or_equal_abs = len(df[df['bic_inferred'] <= df['bic_true']])
    bic_better_or_equal_perc = float(bic_better_or_equal_abs * 100) / len(df)
    bic_worse_abs = len(df[df['bic_inferred'] > df['bic_true']])
    bic_worse_perc = float(bic_worse_abs * 100) / len(df)

    logl_better_or_equal_abs = len(df[df['logl_inferred'] >= df['logl_true']])
    logl_better_or_equal_perc = float(logl_better_or_equal_abs * 100) / len(df)
    logl_worse_abs = len(df[df['logl_inferred'] < df['logl_true']])
    logl_worse_perc = float(logl_worse_abs * 100) / len(df)

    print("Inferred BIC better or equal: " + str(bic_better_or_equal_abs) + ", which is " + "%.2f" % bic_better_or_equal_perc + " %")
    print("Inferred BIC worse: " + str(bic_worse_abs) + ", which is " + "%.2f" % bic_worse_perc + " %")
    print("Inferred logl better or equal: " + str(logl_better_or_equal_abs) + ", which is " + "%.2f" % logl_better_or_equal_perc + " %")
    print("Inferred logl worse: " + str(logl_worse_abs) + ", which is " + "%.2f" % logl_worse_perc + " %")
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle("BIC and Loglikelihood Statistics")
    df['bic_diff'].plot.hist(bins=100, alpha=0.5, title='(bic_true - bic_inferred) / bic_true\n value >0 means inferred BIC was better', ax=axes[0])
    df['logl_diff'].plot.hist(bins=100, alpha=0.5, title='(logl_true - logl_inferred) / logl_true\n value <0 means inferred logl was better', ax=axes[1])
    
    
def reticulation_stats(df):
    reticulations_less_abs = len(df[df['n_reticulations_inferred'] < df['n_reticulations']])
    reticulations_less_perc = float(reticulations_less_abs * 100) / len(df)
    reticulations_equal_abs = len(df[df['n_reticulations_inferred'] == df['n_reticulations']])
    reticulations_equal_perc = float(reticulations_equal_abs * 100) / len(df)
    reticulations_more_abs = len(df[df['n_reticulations_inferred'] > df['n_reticulations']])
    reticulations_more_perc = float(reticulations_more_abs * 100) / len(df)
    print("Inferred n_reticulations less: " + str(reticulations_less_abs) + ", which is " + "%.2f" % reticulations_less_perc + " %")
    print("Inferred n_reticulations equal: " + str(reticulations_equal_abs) + ", which is " + "%.2f" % reticulations_equal_perc + " %")
    print("Inferred n_reticulations more: " + str(reticulations_more_abs) + ", which is " + "%.2f" % reticulations_more_perc + " %")

    
def plot_weirdness_stats(df):
    plt.figure()
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Network Weirdness Statistics")
    df['true_network_weirdness'].plot.hist(bins=10, alpha=0.5, range=(0,1), title='True network weirdness', ax=axes[0])
    df['near_zero_branches_raxml'].plot.hist(bins=10, alpha=0.5, title='Near-zero branches raxml', ax=axes[1])
    plt.tight_layout()
    

def distances(df):
    plt.figure()
    fig, axes = plt.subplots(3, 2, constrained_layout=True)
    fig.suptitle("Topological Network Distances")
    df['hardwired_cluster_distance'].plot.hist(bins=10, alpha=0.5, title='Hardwired cluster distance', ax=axes[0,0])
    df['softwired_cluster_distance'].plot.hist(bins=10, alpha=0.5, title='Softwired cluster distance', ax=axes[0,1])
    df['displayed_trees_distance'].plot.hist(bins=10, alpha=0.5, title='Displayed trees distance', ax=axes[1,0])
    df['tripartition_distance'].plot.hist(bins=10, alpha=0.5, title='Tripartition distance', ax=axes[1,1])
    df['nested_labels_distance'].plot.hist(bins=10, alpha=0.5, title='Nested labels distance', ax=axes[2,0])
    df['path_multiplicity_distance'].plot.hist(bins=10, alpha=0.5, title='Path multiplicity distance', ax=axes[2,1])

    
def plots_setup():
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')


def plot_dataset_size(df):
    plt.figure()
    fig, axes = plt.subplots(1, 3)
    fig.suptitle("Simulated Dataset Stats")
    df['n_taxa'].plot.hist(bins=10, alpha=0.5, title='Number of taxa', ax=axes[0])
    df['near_zero_branches_raxml'].plot.hist(bins=10, alpha=0.5, title='Number of reticulations', ax=axes[1])
    df['msa_size'].plot.hist(bins=10, alpha=0.5, title='MSA size', ax=axes[2])
    plt.tight_layout()


def show_stats(df):
    plot_dataset_size(df)
    plot_weirdness_stats(df)


def show_plots(df):
    bic_logl_stats(df)
    print("")
    reticulation_stats(df)
    print("")
    
    print("")
    distances(df)


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

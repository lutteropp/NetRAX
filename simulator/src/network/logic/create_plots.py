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
    print("Inferred BIC better or equal: " + str(len(df[df['bic_inferred'] <= df['bic_true']])))
    print("Inferred BIC worse: " + str(len(df[df['bic_inferred'] > df['bic_true']])))
    print("Inferred loglh better or equal: " + str(len(df[df['logl_inferred'] >= df['logl_true']])))
    print("Inferred loglh worse: " + str(len(df[df['logl_inferred'] < df['logl_true']])))
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle("BIC and Loglikelihood Statistics")
    df['bic_diff'].plot.hist(bins=100, alpha=0.5, title='(bic_true - bic_inferred) / bic_true\n value >0 means inferred BIC was better', ax=axes[0])
    df['logl_diff'].plot.hist(bins=100, alpha=0.5, title='(logl_true - logl_inferred) / logl_true\n value <0 means inferred logl was better', ax=axes[1])
    
    
def reticulation_stats(df):
    print("Inferred n_reticulations less: " + str(len(df[df['n_reticulations_inferred'] < df['n_reticulations']])))
    print("Inferred n_reticulations equal: " + str(len(df[df['n_reticulations_inferred'] == df['n_reticulations']])))
    print("Inferred n_reticulations more: " + str(len(df[df['n_reticulations_inferred'] > df['n_reticulations']])))

    
def weirdness_stats(df):
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


def show_stats(df):
    print(df.columns)


def show_plots(df):
    plots_setup()
    bic_logl_stats(df)
    print("")
    reticulation_stats(df)
    print("")
    weirdness_stats(df)
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

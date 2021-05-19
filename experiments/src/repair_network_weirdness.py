from netrax_wrapper import check_weird_network
import pandas as pd
import argparse


def recompute_weirdness(prefix):
    input_csv_path = prefix + "_results.csv"
    output_csv_path = prefix + "_results.csv"
    df = pd.read_csv(input_csv_path)
    
    n_equal_tree_pairs = []
    true_network_weirdness = []
        
    for _, row in df.iterrows():
        true_network_path = prefix + "/" + row["true_network_path"]
        n_taxa = int(row['n_taxa'])
        n_pairs, n_equal = check_weird_network(true_network_path, n_taxa)
        
        n_equal_tree_pairs.append(n_equal)
        if n_pairs > 0:
            true_network_weirdness.append(float(n_equal)/n_pairs)
        else:
            true_network_weirdness.append(0.0)
            
        print(row["name"])
        
    df['n_equal_tree_pairs'] = n_equal_tree_pairs
    df['true_network_weirdness'] = true_network_weirdness

    df.to_csv(output_csv_path, index=False)


def parse_command_line_arguments_weirdness():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="")
    args = CLI.parse_args()
    return args.prefix


if __name__ == "__main__":
    prefix = parse_command_line_arguments_weirdness()
    recompute_weirdness(prefix)
    

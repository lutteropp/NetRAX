from netrax_wrapper import network_distance_only
import pandas as pd
import argparse


def append_unrooted_distance(prefix):
    input_csv_path = prefix + "_results.csv"
    output_csv_path = prefix + "_results.csv"
    df = pd.read_csv(input_csv_path)
    unrooted_distance = []
        
    for _, row in df.iterrows():
        true_network_path = prefix + "/" + row["true_network_path"]
        inferred_network_path = prefix + "/" + row["inferred_network_path"]
        n_taxa = int(row["n_taxa"])
        dist = network_distance_only(true_network_path, inferred_network_path, n_taxa)
        unrooted_distance.append(dist)
        print(row["name"])
    df['unrooted_softwired_distance'] = unrooted_distance

    df.to_csv(output_csv_path, index=False)


def parse_command_line_arguments_unrooted_distance():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="")
    args = CLI.parse_args()
    return args.prefix


if __name__ == "__main__":
    prefix = parse_command_line_arguments_unrooted_distance()
    append_unrooted_distance(prefix)

from dendroscope_wrapper import evaluate
import pandas as pd
import argparse

TOPOLOGICAL_DISTANCE_NAMES = ['hardwired_cluster_distance', 'softwired_cluster_distance',
                              'displayed_trees_distance', 'tripartition_distance', 'nested_labels_distance', 'path_multiplicity_distance']


def append_distances(prefix):
    input_csv_path = prefix + "_results.csv"
    output_csv_path = prefix + "_results.csv"
    df = pd.read_csv(input_csv_path)
    extra_cols = {}
    for name in TOPOLOGICAL_DISTANCE_NAMES:
        extra_cols[name] = []
        
    for _, row in df.iterrows():
        true_network_path = prefix + "/" + row["true_network_path"]
        inferred_network_path = prefix + "/" + row["inferred_network_path"]
        topological_distances = evaluate(
            true_network_path, inferred_network_path)
        for name in TOPOLOGICAL_DISTANCE_NAMES:
            extra_cols[name].append(topological_distances[name])
        print(row["name"])
    for name in TOPOLOGICAL_DISTANCE_NAMES:
        df[name] = extra_cols[name]

    df.to_csv(output_csv_path, index=False)


def parse_command_line_arguments_distances():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="")
    args = CLI.parse_args()
    return args.prefix


if __name__ == "__main__":
    prefix = parse_command_line_arguments_distances()
    append_distances(prefix)

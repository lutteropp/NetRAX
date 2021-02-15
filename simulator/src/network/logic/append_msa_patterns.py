from dendroscope_wrapper import evaluate
import pandas as pd
import argparse


def append_patterns(prefix):
    input_csv_path = prefix + "_results.csv"
    output_csv_path = prefix + "_results.csv"
    df = pd.read_csv(input_csv_path)
    patterns = []
        
    for _, row in df.iterrows():
        raxml_log_path = prefix + "/" + row["name"] + ".raxml.log"
        num_patterns = -1
        for line in open(raxml_log_path).readlines():
            if line.startswith("Alignment comprises"):
                num_patterns = int(line.split(' ')[5])
                break
        patterns.append(num_patterns)
        print(row["name"])

    df['msa_patterns'] = patterns
    df.to_csv(output_csv_path, index=False)


def parse_command_line_arguments_patterns():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="")
    args = CLI.parse_args()
    return args.prefix


if __name__ == "__main__":
    prefix = parse_command_line_arguments_patterns()
    append_patterns(prefix)

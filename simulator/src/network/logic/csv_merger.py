import pandas as pd
import argparse


def merge_csvs(inpaths, outpath):
   combined_csv = pd.concat([pd.read_csv(f) for f in inpaths ])
   combined_csv.to_csv(outpath, index=False, encoding='utf-8-sig')


def postprocess_merge(iterations, prefix):
    local_csv_paths = []
    for it in range(iterations):
        local_prefix = prefix + "_" + str(it)
        local_csv_paths.append(local_prefix + "_results.csv")
    merge_csvs(local_csv_paths, prefix + "_results.csv")


def parse_command_line_arguments_postprocess():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix",nargs=1, type=str, default="small_network")
    CLI.add_argument("--iterations", nargs=1, type=int, default=1)
    args = CLI.parse_args()
    return args.prefix, args.iterations


if __name__ == 'main':
    iterations, prefix = parse_command_line_arguments_postprocess()
    postprocess_merge(iterations, prefix)

import pandas as pd
import argparse
import os.path


def merge_csvs(inpaths, outpath):
    real_inpaths = [p for p in inpaths if os.path.isfile(p)]
    for p in inpaths:
        if not os.path.isfile(p):
            print("Not a file: " + p)
            #raise Exception("Not a file")
    combined_csv = pd.concat([pd.read_csv(f) for f in real_inpaths])
    combined_csv.to_csv(outpath, index=False, encoding='utf-8-sig')


def postprocess_merge(prefix, iterations):
    local_csv_paths = []
    for it in range(iterations):
        local_prefix = prefix + "_" + str(it)
        local_csv_paths.append(local_prefix + "_results.csv")
    merge_csvs(local_csv_paths, prefix + "_results.csv")


def parse_command_line_arguments_postprocess():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="small_network")
    CLI.add_argument("--iterations", type=int, default=1)
    args = CLI.parse_args()
    return args.prefix, args.iterations


if __name__ == '__main__':
    prefix, iterations = parse_command_line_arguments_postprocess()
    postprocess_merge(prefix, iterations)

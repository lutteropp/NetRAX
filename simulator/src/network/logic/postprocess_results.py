from append_topologial_distances import append_distances_netrax
from append_msa_patterns import append_patterns
from csv_merger import postprocess_merge

import argparse


def parse_command_line_arguments_postprocess():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--prefix", type=str, default="small_network")
    CLI.add_argument("--iterations_global", type=int, default=8)
    CLI.add_argument("--iterations_local", type=int, default=16)
    args = CLI.parse_args()
    return args.prefix, args.iterations_global, args.iterations_local


if __name__ == "__main__":
    prefix, iterations_global, iterations_local = parse_command_line_arguments_postprocess()
    for g_it in range(iterations_global):
        postprocess_merge(prefix + "_" + str(g_it), iterations_local)
    postprocess_merge(prefix, iterations_global)
    append_patterns(prefix)
    append_distances_netrax(prefix)

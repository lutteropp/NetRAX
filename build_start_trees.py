import subprocess
import argparse
import os
import math

NETRAX_CORE_PATH = "/home/luttersh/NetRAX/bin/netrax"
#NETRAX_CORE_PATH = "mpiexec /home/sarah/code-workspace/NetRAX/bin/netrax"

RAXML_PATH = "/home/luttersh/NetRAX/experiments/deps/raxml-ng"
#RAXML_PATH = "/home/sarah/code-workspace/NetRAX/experiments/deps/raxml-ng"


def run_raxml(msa_path, partitions_path, seed, start_trees_output_path, no_inference, num_parsimony_trees, num_random_trees):
    raxml_cmd = RAXML_PATH
    if no_inference:
        raxml_cmd + " --start"
    else:
        raxml_cmd += " --search"
    raxml_cmd += "--tree pars{" + str(num_parsimony_trees) + "},rand{" + str(num_random_trees) + "}"
    raxml_cmd += " --msa " + msa_path + " --model " + partitions_path
    raxml_cmd += " --seed " + str(seed)
    raxml_cmd += " --redo"
    raxml_cmd += " --prefix " + start_trees_output_path

    print(raxml_cmd)
    p = subprocess.run(raxml_cmd.split(), stdout=subprocess.PIPE, check=True)


def find_unique_trees(trees_file):
    raxml_cmd = RAXML_PATH + " --rfdist --tree " + trees_file + "--prefix RF"
    print(raxml_cmd)
    p = subprocess.run(raxml_cmd.split(), stdout=subprocess.PIPE, check=True)


def build_trees(msa_path, partitions_path, seed, start_trees_output_path, no_inference, keep_only_unique, num_parsimony_trees, num_random_trees):
    run_raxml(msa_path, partitions_path, seed, start_trees_output_path, no_inference, num_parsimony_trees, num_random_trees)
    trees = []
    with open(start_trees_output_path + ".raxml.startTrees") as f:
        trees = f.readlines()
        f.close()

    if keep_only_unique:
        trees = find_unique_trees(start_trees_output_path + ".raxml.startTrees")
    with open(start_trees_output_path, 'w') as g:
        for tree in trees:
            if len(tree.strip()) > 0:
                g.write(tree.strip() + "\n")
        g.write(trees)
        g.close()


def parse_command_line_arguments_build_start_trees():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--msa_path", type=str)
    CLI.add_argument("--partitions_path", type=str)
    CLI.add_argument("--seed", type=int, default=0)
    CLI.add_argument("--start_trees_output_path", type=str)
    CLI.add_argument("--no_inference", action='store_true')
    CLI.add_argument("--keep_only_unique", action='store_true')
    CLI.add_argument("--num_parsimony_trees", type=int, default=10)
    CLI.add_argument("--num_random_trees", type=int, default=10)
    args = CLI.parse_args()
    return args.msa_path, args.partitions_path, args.seed, args.start_trees_output_path, args.no_inference, args.keep_only_unique, args.num_parsimony_trees, args.num_random_trees


if __name__ == '__main__':
    msa_path, partitions_path, seed, start_trees_output_path, no_inference, keep_only_unique, num_parsimony_trees, num_random_trees = parse_command_line_arguments_build_start_trees()
    build_trees(msa_path, partitions_path, seed, start_trees_output_path, no_inference, keep_only_unique, num_parsimony_trees, num_random_trees)
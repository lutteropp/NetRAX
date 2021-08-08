import subprocess
import argparse
import os
import math

NETRAX_CORE_PATH = "/home/luttersh/NetRAX/bin/netrax"
#NETRAX_CORE_PATH = "/home/sarah/code-workspace/NetRAX/bin/netrax"

RAXML_PATH = "/home/luttersh/NetRAX/experiments/deps/raxml-ng"
#RAXML_PATH = "/home/sarah/code-workspace/NetRAX/experiments/deps/raxml-ng"


#from https://www.endpoint.com/blog/2015/01/getting-realtime-output-using-python/
def run_command(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    timeout=0
    full_output = []
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            line = output.strip().decode()
            full_output.append(line)
            print(line)
            timeout=0
        else:
            timeout+=1
        if timeout >= 100:
            break
    return full_output



def run_raxml(msa_path, partitions_path, seed, name, no_inference, num_parsimony_trees, num_random_trees):
    raxml_cmd = RAXML_PATH
    if no_inference:
        raxml_cmd += " --start"
    else:
        raxml_cmd += " --search"
    raxml_cmd += " --tree pars{" + str(num_parsimony_trees) + "},rand{" + str(num_random_trees) + "}"
    raxml_cmd += " --msa " + msa_path + " --model " + partitions_path
    raxml_cmd += " --seed " + str(seed)
    raxml_cmd += " --redo"
    raxml_cmd += " --prefix " + name

    print(raxml_cmd)
    run_command(raxml_cmd)
    #p = subprocess.run(raxml_cmd.split(), stdout=subprocess.PIPE, check=True)


def find_unique_trees(trees_file, output_path):
    rf_dist_file = output_path + ".raxml.rfDistances"
    raxml_cmd = RAXML_PATH + " --rfdist --tree " + trees_file + " --prefix " + output_path + " --redo"
    print(raxml_cmd)
    run_command(raxml_cmd)
    #subprocess.run(raxml_cmd.split(), stdout=subprocess.PIPE, check=True)
    lines = open(rf_dist_file).readlines()
    os.remove(rf_dist_file)
    
    pairwise_rf_dist = []
    bad_trees = set()
    for line in lines:
        if len(line.strip()) > 0:
            splitted = line.strip().split("\t")
            id1 = int(splitted[0])
            id2 = int(splitted[1])
            dist = int(splitted[2])
            if id1 not in bad_trees and id2 not in bad_trees:
                if dist == 0:
                    bad_trees.add(min(id1, id2))
    good_trees = []

    with open(trees_file) as f:
        trees = f.readlines()
        f.close()

    for i in range(len(trees)):
        if i not in bad_trees:
            good_trees.append(trees[i])

    n_trees = 0
    with open(output_path, 'w') as g:
        for tree in trees:
            if len(tree.strip()) > 0:
                g.write(tree.strip() + "\n")
                n_trees += 1
        g.close()
    print("\n" + str(n_trees) + " start trees for NetRAX written to: " + output_path)
    return good_trees

def build_trees(msa_path, partitions_path, seed, name, no_inference, num_parsimony_trees, num_random_trees):
    trees_path_no_inference = name + ".raxml.startTree"
    trees_path_only_best = name + ".raxml.bestTree"
    trees_path_all_ml = name + ".raxml.mlTrees"
    trees_path_no_inference_unique = name + ".raxml.startTree.unique"
    trees_path_all_ml_unique = name + ".raxml.mlTrees.unique"

    run_raxml(msa_path, partitions_path, seed, name, no_inference, num_parsimony_trees, num_random_trees)

    find_unique_trees(trees_path_no_inference, trees_path_no_inference_unique)
    if not no_inference:
        find_unique_trees(trees_path_all_ml, trees_path_all_ml_unique)    

def parse_command_line_arguments_build_start_trees():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--msa_path", type=str)
    CLI.add_argument("--partitions_path", type=str)
    CLI.add_argument("--seed", type=int, default=0)
    CLI.add_argument("--name", type=str)
    CLI.add_argument("--no_inference", action='store_true')
    CLI.add_argument("--num_parsimony_trees", type=int, default=10)
    CLI.add_argument("--num_random_trees", type=int, default=10)
    args = CLI.parse_args()
    return args.msa_path, args.partitions_path, args.seed, args.name, args.no_inference, args.num_parsimony_trees, args.num_random_trees


if __name__ == '__main__':
    msa_path, partitions_path, seed, name, no_inference, num_parsimony_trees, num_random_trees = parse_command_line_arguments_build_start_trees()
    build_trees(msa_path, partitions_path, seed, name, no_inference, num_parsimony_trees, num_random_trees)

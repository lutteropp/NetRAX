import subprocess
import argparse
import os
import math

NETRAX_CORE_PATH = "/home/luttersh/NetRAX/bin/netrax"
#NETRAX_CORE_PATH = "mpiexec /home/sarah/code-workspace/NetRAX/bin/netrax"


#from https://www.endpoint.com/blog/2015/01/getting-realtime-output-using-python/
def run_command(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip().decode())
    return process.stdout.decode()


def infer_network(start_network_path, msa_path, partitions_path, likelihood_type, brlen_linkage_type, seed, inferred_network_path, is_good_start, logfile_path):
    netrax_cmd = "mpiexec " + NETRAX_CORE_PATH + " --start_network " + \
        start_network_path + " --msa " + msa_path + " --model " + partitions_path + \
        " --brlen " + brlen_linkage_type + " --output " + inferred_network_path
    if likelihood_type == "average":
        netrax_cmd += " --average_displayed_tree_variant"
    elif likelihood_type == "best":
        netrax_cmd += " --best_displayed_tree_variant"
    if seed > 0:
        netrax_cmd += " --seed " + str(seed)
    if is_good_start:
        netrax_cmd += " --good_start"
    print(netrax_cmd)
    cmd_output = run_command(netrax_cmd)

    with open(logfile_path, 'w') as logfile:
        logfile.write(cmd_output)
        logfile.close()

    bic = 0
    aic = 0
    aicc = 0
    logl = 0
    n_reticulations = 0
    runtime_in_seconds = 0
    network = ""
    netrax_output = cmd_output.splitlines()
    for line in netrax_output:
        if line.startswith("Number of reticulations:"):
            n_reticulations = float(line.split(": ")[1])
        if line.startswith("BIC Score:"):
            bic = float(line.split(": ")[1])
        if line.startswith("Loglikelihood:"):
            logl = float(line.split(": ")[1])
        if line.startswith("AIC Score:"):
            aic = float(line.split(": ")[1])
        if line.startswith("AICc Score:"):
            aicc = float(line.split(": ")[1])
        if line.startswith("Total runtime:"):
            runtime_in_seconds = float(line.split(": ")[1].split(" ")[0])
    newick = open(inferred_network_path).read().strip()
    return (bic, aic, aicc, logl, n_reticulations, runtime_in_seconds, newick)


def run_netrax_multi(name, msa_path, partitions_path, likelihood_type, brlen_linkage_type, seed, start_networks, is_good_start):
    networks = open(start_networks).readlines()
    inferred_network_path = name + "_inferred_network.nw"
    os.mkdir(name + "_subruns")
    results = []
    for i in range(len(networks)):
        newick = networks[i].strip()
        if len(newick) == 0:
            continue
        start_network_path = name + "_subruns/run_" + str(i) + "_start_network.nw"
        run_inferred_network_path = name + "_subruns/run_" + str(i) + "_inferred_network.nw"
        logfile_path = name + "_subruns/run_" + str(i) + "_logfile.txt"
        with open(start_network_path, 'w') as f:
            f.write(newick + "\n")
            f.close()
        results.append(infer_network(start_network_path, msa_path, partitions_path, likelihood_type, brlen_linkage_type, seed, run_inferred_network_path, is_good_start, logfile_path))
    best_bic = math.inf
    best_aic = math.inf
    best_aicc = math.inf
    best_logl = -math.inf
    total_runtime_in_seconds = 0
    best_network = ""
    best_n_reticulations = 0
    for (bic, aic, aicc, logl, n_reticulations, runtime_in_seconds, newick) in results:
        if bic < best_bic:
            best_bic = bic
            best_aic = aic
            best_aicc = aicc
            best_logl = logl
            best_n_reticulations = n_reticulations
            best_network = newick
        total_runtime_in_seconds += runtime_in_seconds
    with open(inferred_network_path, 'w') as f:
        f.write(best_network + "\n")
        f.close()
    print("Total inference runtime: " + str(total_runtime_in_seconds) + " seconds.")
    print("Best inferred network has" + str(best_n_reticulations) + " reticulations.")
    print("Best inferred network has BIC: " + str(best_bic))
    print("Best inferred network has AIC: " + str(best_aic))
    print("Best inferred network has cAIC: " + str(best_aicc))
    print("Best inferred network has logl: " + str(best_logl))
    print("Best inferred network is:\n" + best_network)

def parse_command_line_arguments_netrax_multi():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--name", type=str)
    CLI.add_argument("--msa_path", type=str)
    CLI.add_argument("--partitions_path", type=str)
    CLI.add_argument("--likelihood_type", type=str)
    CLI.add_argument("--brlen_linkage_type", type=str, default="linked")
    CLI.add_argument("--seed", type=int, default=0)
    CLI.add_argument("--start_networks", type=str)
    CLI.add_argument("--good_start", action='store_true')
    args = CLI.parse_args()
    return args.name, args.msa_path, args.partitions_path, args.likelihood_type, args.brlen_linkage_type, args.seed, args.start_networks, args.good_start


if __name__ == '__main__':
    name, msa_path, partitions_path, likelihood_type, brlen_linkage_type, seed, start_networks, good_start = parse_command_line_arguments_netrax_multi()
    run_netrax_multi(name, msa_path, partitions_path, likelihood_type, brlen_linkage_type, seed, start_networks, good_start)
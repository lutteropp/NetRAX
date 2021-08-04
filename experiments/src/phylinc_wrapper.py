import subprocess
import argparse

def run_phylinc_inference(start_network, msa_file, max_reticulations):
    cmd = "julia run_phylinc_script.jl " + start_network + " " + msa_file + " " + str(max_reticulations)
    print(cmd, flush=True)

    p = subprocess.run(cmd.split(), stdout=subprocess.PIPE, check=True)
    cmd_output = p.stdout.decode()
    print(cmd_output)
    phylinc_output = cmd_output.splitlines()
    topo_cnt = 0
    inferred_network = ""
    runtime_cnt = 0
    runtime = 0
    for i in range(len(phylinc_output)):
        line = phylinc_output[i]
        if topo_cnt == 2:
            inferred_network = line
            topo_cnt = 9999
        if line.startswith("Best topology"):
            topo_cnt += 1
        elif line.startswith("Total time elapsed"):
            runtime_cnt += 1
            if runtime_cnt == 2:
                runtime = line.split(": ")[1].split(" seconds")[0]
    return inferred_network, float(runtime)


def parse_command_line_arguments_phylinc():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--start_network", type=str)
    CLI.add_argument("--msa", type=str)
    CLI.add_argument("--max_reticulations", type=int, default=5)
    args = CLI.parse_args()
    return args.start_network, args.msa, args.max_reticulations


if __name__ == '__main__':
    start_network, msa_file, max_reticulations = parse_command_line_arguments_phylinc()
    inferred_network, runtime = run_phylinc_inference(start_network, msa_file, max_reticulations)
    print(inferred_network)
    print(str(runtime) + " seconds.")
import subprocess
from experiment_model import Dataset

RAXML_PATH = 'raxml-ng'


def infer_raxml_tree(dataset):
    cmd = RAXML_PATH + ' --msa ' + str(dataset.msa_path) + ' --model ' + str(dataset.partitions_path) + ' --prefix ' + str(dataset.name) + ' --seed 42'
    print(cmd)
    cmd_status, cmd_output = subprocess.getstatusoutput(cmd)
    print(cmd_output)
    if cmd_status != 0:
        raise Exception("Inferring raxml tree failed")
    lines = cmd_output.splitlines()
    near_zero_branches = 0
    for line in lines:
        if "near-zero branches" in line and "WARNING: Best ML tree contains" in line:
            near_zero_branches = int(line.split('WARNING: Best ML tree contains ')[1].split(' ')[0])
    return near_zero_branches
    
    
def compute_rf_dist(tree_1_path, tree_2_path):
    cmd = RAXML_PATH + " --rf " + tree_1_path + "," + tree_2_path
    print(cmd)
    cmd_status, cmd_output = subprocess.getstatusoutput(cmd)
    print(cmd_output)
    if cmd_status != 0:
        raise Exception("Compute RF dist failed")
    lines = cmd_output.splitlines()
    rf_abs = -1
    rf_rel = -1
    for line in lines:
        if line.startswith("Average absolute RF distance"):
            rf_abs = float(line.split(": ")[1])
        elif line.startswith("Average relative RF distance"):
            rf_rel = float(line.split(": ")[1])
    return (rf_abs, rf_rel)

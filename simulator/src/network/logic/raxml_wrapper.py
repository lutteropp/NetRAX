import subprocess
from experiment_model import Dataset

RAXML_PATH = 'raxml-ng'


def infer_raxml_tree(dataset):
    cmd = RAXML_PATH + ' --msa ' + str(dataset.msa_path) + ' --model ' + str(dataset.partitions_path) + ' --prefix ' + str(dataset.name) + ' --seed 42'
    print(cmd)
    print(subprocess.getoutput(cmd))
    
    
def compute_rf_dist(tree_1_path, tree_2_path):
    cmd = RAXML_PATH + " --rf " + tree_1_path + "," + tree_2_path
    print(cmd)
    lines = subprocess.getoutput(cmd).splitlines()
    print(lines)
    rf_abs = -1
    rf_rel = -1
    for line in lines:
        if line.startswith("Average absolute RF distance"):
            rf_abs = float(line.split(": ")[1])
        elif line.startswith("Average relative RF distance"):
            rf_rel = float(line.split(": ")[1])
    return (rf_abs, rf_rel)

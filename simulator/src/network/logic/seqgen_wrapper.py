import subprocess
from run_experiments import Dataset

SEQGEN_PATH = "seq-gen"


def simulate_msa(dataset):
    cmd = 'seq-gen -mHKY -t3.0 -f0.3,0.2,0.2,0.3 -l' + str(dataset.msa_size)+'-p' + str(dataset.n_trees)+' < ' + dataset.extracted_trees_path + ' > ' + dataset.msa_path
    subprocess.getoutput(cmd)
    

import subprocess
import collections

from experiment_model import Dataset

SEQGEN_PATH = 'seq-gen'


def simulate_msa(dataset):
    cmd = SEQGEN_PATH + ' -mHKY -t3.0 -f0.3,0.2,0.2,0.3 -l' + str(dataset.msa_size)+' -p' + str(dataset.n_trees)+' < ' + dataset.extracted_trees_path + ' > ' + dataset.msa_path
    print(cmd)
    print(subprocess.getoutput(cmd))

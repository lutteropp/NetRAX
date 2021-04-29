import subprocess
import collections

from experiment_model import Dataset

SEQGEN_PATH = './seq-gen'


def simulate_msa(dataset):
    cmd = SEQGEN_PATH + ' ' + dataset.seqgen_params + ' -l' + str(dataset.msa_size)+' -p' + str(
        dataset.n_trees)+' < ' + dataset.extracted_trees_path + ' > ' + dataset.msa_path
    print(cmd, flush=True)
    p = subprocess.Popen(cmd)
    cmd_output, _ = p.communicate()
    cmd_status = p.returncode     
    print(cmd_output)
    if cmd_status != 0:
        raise Exception("Simulate MSA failed")

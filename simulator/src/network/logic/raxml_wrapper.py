import subprocess
from experiment_model import Dataset

RAXML_PATH = 'raxml-ng'


def infer_raxml_tree(dataset):
    cmd = RAXML_PATH + ' --msa ' + str(dataset.msa_path) + ' --model ' + str(dataset.partitions_path) + ' --prefix ' + str(dataset.name) + ' --seed 42'
    print(cmd)
    subprocess.getoutput(cmd)

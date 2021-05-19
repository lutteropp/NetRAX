import subprocess
import collections

from experiment_model import Dataset

SEQGEN_PATH = '../deps/seq-gen'


def simulate_msa(dataset):
    cmd = SEQGEN_PATH + ' ' + dataset.seqgen_params + ' -l' + str(dataset.msa_size)+' -p' + str(dataset.n_trees)
    print(cmd, flush=True)
    infile = open(dataset.extracted_trees_path)
    outfile = open(dataset.msa_path, "w")

    p = subprocess.run(cmd.split(), stdin=infile, stdout=outfile, check=True)
    outfile.close()
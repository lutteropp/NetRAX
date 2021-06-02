
import subprocess
import random
import sys

from Bio import AlignIO

"""

Author: Sarah Lutteropp
A script that subsamples a partitioned MSA, trying to find a smaller dataset that already leads to a bug.

"""

ORIG_MSA = "data/datasets_40t_4r_small/0_0_msa.txt"
ORIG_PARTITIONS = "data/datasets_40t_4r_small/0_0_partitions.txt"
ORIG_OUTPUT = "data/datasets_40t_4r_small/0_0_BEST_LINKED_FROM_RAXML_inferred_network.nw"

MIN_MSA_SIZE = 100
MIN_N_TAXA = 4

def build_command(msa_path, partitions_path, output_path):
    return "mpiexec ../bin/netrax --msa " + msa_path + " --output " + output_path + " --model " + partitions_path + " --best_displayed_tree_variant --num_random_start_networks 0 --num_parsimony_start_networks 1 --brlen linked --seed 42"

def orig_command():
    return build_command(ORIG_MSA, ORIG_PARTITIONS, ORIG_OUTPUT)

def trimmed_seq(orig_seq, deleted_cols):
    l = list(orig_seq)
    for c in deleted_cols:
        l[c-1] = ""
    return "".join(l)

def write_msa(taxon_names, msa, msa_path, deleted_rows=[], deleted_cols=[]):
    with open(msa_path, 'w') as f:
        for i in range(len(taxon_names)):
            if i in deleted_rows:
                continue
            f.write(">" + taxon_names[i] + "\n")
            f.write(trimmed_seq(msa[i], deleted_cols) + "\n")
        f.close()

def parse_msa(msa_path):
    taxon_names = []
    msa = []
    alignment = AlignIO.read(open(msa_path), "phylip")
    for record in alignment:
        taxon_names.append(record.id)
        msa.append(record.seq)
    return (taxon_names, msa)

def trimmed_ranges(orig_ranges, deleted_cols):
    trimmed_ranges = []
    deleted_cols.sort()
    for r in range(len(orig_ranges)):
        current_start = orig_ranges[r][0]
        current_end = orig_ranges[r][1]
        # we need to know how many columns have been deleted before current start
        n_before_start = len([c for c in deleted_cols if c < current_start])
        # and we need to know how many columns have been deleted before current end
        n_before_end = len([c for c in deleted_cols if c <= current_end])
        # this gives us the new partition range
        trimmed_ranges.append((current_start - n_before_start, current_end - n_before_end))
    return trimmed_ranges

def write_partitions(model, name, prange, partitions_path):
    with open(partitions_path, 'w') as f:
        for i in range(len(model)):
            if (prange[i][0] <= prange[i][1]):
                f.write(model[i] + "," + name[i] + "=" + str(prange[i][0]) + "-" + str(prange[i][1]) + "\n")
        f.close()

def parse_partitions(partitions_path):
    model = []
    name = []
    prange = []
    with open(partitions_path) as f:
        lines = f.readlines()
        for line in lines:
            model.append(line.split(',')[0])
            name.append(line.split(',')[1].split('=')[0])
            prange_start = int(line.split(',')[1].split('=')[1].split('-')[0])
            prange_end = int(line.split(',')[1].split('=')[1].split('-')[1])
            prange.append((prange_start, prange_end))
    return (model, name, prange)

def write_data(taxon_names, msa, model, name, prange, deleted_rows, deleted_cols, msa_path, partitions_path):
    write_msa(taxon_names, msa, msa_path, deleted_rows, deleted_cols)
    write_partitions(model, name, trimmed_ranges(prange, deleted_cols), partitions_path)
    
def run_command(cmd):
    print(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    for line in p.stdout:
        print(line.decode(), end='')
    p.wait()
    retcode = p.returncode

    print("retcode: " + str(retcode))
    if (retcode != 0):
        print("Found a bug!!! Use this command to reproduce:")
        print(cmd)
        sys.exit()

    return retcode

def run_on_subsampled_data(taxon_names, msa, model, name, prange, deleted_rows, deleted_cols, msa_path, partitions_path, output_path):
    n_taxa = len(taxon_names)
    n_cols = prange[-1][1]
    print("Doing subsampled run on " + str(n_taxa - len(deleted_rows)) + " taxa and " + str(n_cols - len(deleted_cols)) + " sites")
    write_data(taxon_names, msa, model, name, prange, deleted_rows, deleted_cols, msa_path, partitions_path)
    cmd = build_command(msa_path, partitions_path, output_path)
    return run_command(cmd)

def subsample(n_taxa, n_cols, fraction_taxa, fraction_cols):
    n_del_taxa = int(n_taxa - float(n_taxa) * fraction_taxa)
    n_del_cols = int(n_cols - float(n_cols) * fraction_cols)

    # rows are indexed starting from 0, cols are indexed starting from 1 here!
    all_rows = [i for i in range(n_taxa)]
    all_cols = [i+1 for i in range(n_cols)]
    deleted_rows = random.sample(all_rows, n_del_taxa)
    deleted_cols = random.sample(all_cols, n_del_cols)
    return deleted_rows, deleted_cols

def search_bug_step(taxon_names, msa, model, name, prange, fraction_taxa, fraction_cols, it):
    n_taxa = len(taxon_names)
    n_cols = prange[-1][1]

    n_subsampled_taxa = int(float(n_taxa) * fraction_taxa)
    n_subsampled_cols = int(float(n_cols) * fraction_cols)

    identifier = str(n_subsampled_taxa) + "_" + str(n_subsampled_cols) + "_" + str(it)
    msa_path = "sampled_msa_" + identifier + ".fasta"
    partitions_path = "sampled_partitions_" + identifier + ".txt"
    output_path = "sampled_output_" + identifier + ".txt"

    deleted_rows, deleted_cols = subsample(n_taxa, n_cols, fraction_taxa, fraction_cols)

    return run_on_subsampled_data(taxon_names, msa, model, name, prange, deleted_rows, deleted_cols, msa_path, partitions_path, output_path)

def search_bug(taxon_names, msa, model, name, prange):
    n_taxa = len(taxon_names)
    n_cols = prange[-1][1]
    taxon_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    msa_fractions = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    iterations = 3

    retcode = 0

    for fraction_taxa in taxon_fractions:
        n_subsampled_taxa = int(float(n_taxa) * fraction_taxa)
        if n_subsampled_taxa < MIN_N_TAXA:
            continue
        for fraction_cols in msa_fractions:
            n_subsampled_cols = int(float(n_cols) * fraction_cols)
            if n_subsampled_cols < MIN_MSA_SIZE:
                 continue
            for it in range(iterations):
                retcode = search_bug_step(taxon_names, msa, model, name, prange, fraction_taxa, fraction_cols, it)
                if retcode !=0:
                    break
        if retcode != 0:
            print("We think a bug was found")
            break

if __name__ == "__main__":
    print("original command:")
    print(orig_command() + "\n")
    taxon_names, msa = parse_msa(ORIG_MSA)
    model, name, prange = parse_partitions(ORIG_PARTITIONS)
    search_bug(taxon_names, msa, model, name, prange)

import subprocess

ORIG_MSA = "data/datasets_40t_4r_small/0_0_msa.txt"
ORIG_PARTITIONS = "data/datasets_40t_4r_small/0_0_partitions.txt"
ORIG_OUTPUT = "data/datasets_40t_4r_small/0_0_BEST_LINKED_FROM_RAXML_inferred_network.nw"

def build_command(msa_path, partitions_path, output_path):
    return "mpiexec ../bin/netrax --msa " + msa_path + " --output " + output_path + " --model " + partitions_path + " --best_displayed_tree_variant --num_random_start_networks 0 --num_parsimony_start_networks 1 --brlen linked --seed 42"

def orig_command():
    return build_command(ORIG_MSA, ORIG_PARTITIONS, ORIG_OUTPUT)

def trimmed_seq(orig_seq, deleted_cols):
    l = list(orig_seq)
    for c in deleted_cols:
        l[c] = ""
    return "".join(l)

def write_msa(taxon_names, msa, msa_path, deleted_rows=[], deleted_cols=[]):
    with open(msa_path, 'w') as f:
        for i in range(len(taxon_names)):
            if i in deleted_rows:
                continue
            f.write(taxon_names[i] + "\n")
            f.write(trimmed_seq(msa[i], deleted_cols) + "\n")
        f.close()

def parse_msa(msa_path):
    # assumes fasta format for now
    taxon_names = []
    msa = []
    with open(msa_path) as f:
        lines = f.readlines.split()
        for i in range(len(lines)):
            if i % 2 == 0:
                taxon_names.append(lines[i])
            else:
                msa.append(lines[i])
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
        trimmed_ranges.append(current_start - n_before_start, current_end - n_before_end)
    return trimmed_ranges

def write_partitions(model, name, range, partitions_path):
    with open(partitions_path, 'w') as f:
        for i in range(len(model)):
            if (range[i][0] <= range[i][1]):
                f.write(model[i] + "," + name[i] + "=" + str(range[i][0]) + "-" + str(range[i][1]))
        f.close()

def parse_partitions(partitions_path):
    model = []
    name = []
    range = []
    with open(partitions_path) as f:
        lines = f.readlines().split()
        for line in lines:
            model.append(line.split(',')[0])
            name.append(line.split(',')[1].split('=')[0])
            range.append(int(line.split(',')[1].split('=')[1].split('-')[0]), int(line.split(',')[1].split('=')[1].split('-')[1]))
    return (model, name, range)

def write_data(taxon_names, msa, model, name, range, deleted_rows, deleted_cols, msa_path, partitions_path):
    write_msa(taxon_names, msa, msa_path, deleted_rows, deleted_cols)
    write_partitions(model, name, trimmed_ranges(range, deleted_cols), partitions_path)
    
def run_command(cmd):
    print(cmd)
    p = subprocess.run(cmd.split(), stdout=subprocess.PIPE, check=True)
    retcode = p.returncode
    if (retcode != 0):
        cmd_output = p.stdout.decode()
        print(cmd_output)
    return (retcode == 0)

def run_on_subsampled_data(taxon_names, msa, model, name, range, deleted_rows, deleted_cols, msa_path, partitions_path, output_path):
    write_data(taxon_names, msa, model, name, range, deleted_rows, deleted_cols, msa_path, partitions_path)
    cmd = build_command(msa_path, partitions_path, output_path)
    return run_command(cmd)
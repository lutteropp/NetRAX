import argparse
import random

def parse_orig_partitions(partitions_path):
    model = []
    name = []
    psites = []
    with open(partitions_path) as f:
        lines = f.readlines()
        for line in lines:
            model.append(line.split(',')[0])
            name.append(line.split(',')[1].split('=')[0])
            prange_start = int(line.split(',')[1].split('=')[1].split('-')[0])
            prange_end = int(line.split(',')[1].split('=')[1].split('-')[1])
            psites.append([i for i in range(prange_start, prange_end+1)])
    return model, name, psites


def write_partitions(model, name, psites, partitions_path):
    with open(partitions_path, 'w') as f:
        for i in range(len(model)):
            if (len(psites[i] > 0)):
                f.write(model[i] + "," + name[i] + "=" + ",".join(psites[i]) + "\n")
        f.close()


def scramble_partitions(infile_name, outfile_name, factor):
    (model, name, psites) = parse_orig_partitions(infile_name)
    part_count = len(name)
    if part_count == 1:
        return model, name, psites
    removed_sites = [[] for i in range(part_count)]
    extra_sites = [[] for i in range(part_count)]
    # fill removed_sites and extra_sites
    for i in range(part_count):
        n_shuffle = int(len(psites[i]) * factor)
        removed_sites[i] = random.sample(psites[i], n_shuffle)
        for j in removed_sites[i]:
            new_part = random.randint(0, part_count - 1)
            while new_part == i:
                new_part = random.randint(0, part_count - 1)
            extra_sites[new_part].append(j)

    new_psites = [[] for i in range(part_count)]
    # fill new_psites
    for i in range(part_count):
        for j in psites[i]:
            if not j in removed_sites[i]:
                new_psites[i].append(j)
        for j in extra_sites[i]:
            new_psites[i].append(j)
    return model, name, new_psites


def parse_command_line_arguments_pscramble():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--infile", type=str, default="partitions.txt")
    CLI.add_argument("--outfile", type=str, default="scrambled_partitions.txt")
    CLI.add_argument("--fraction", type=float, default=0.1)
    args = CLI.parse_args()
    return args.infile, args.outfile, args.fraction


if __name__ == '__main__':
    infile_name, outfile_name, fraction = parse_command_line_arguments_pscramble()
    model, name, new_psites = scramble_partitions(infile_name, outfile_name, fraction)
    write_partitions(model, name, new_psites, outfile_name)

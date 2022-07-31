#!/usr/bin/python
import sys

# Takes a string in Extended NEWICK Format, and drops the reticulation probabilities and support scores, keeping only the branch length.
def convert_newick_to_dendroscope(newick):
    new_newick = ""
    # drop probs and support scores
    seenColon = False
    skip = False
    for c in newick:
        if c == ':':
            skip = seenColon
            seenColon = True
        elif c in [',', '(', ')', ';']:
            skip = False
            seenColon = False
        if not skip and c != '\n':
            new_newick += c
    # ensure reticulations are names #H0 instead of #0 etc.
    new_newick_fixed_names = ""
    prev = ''
    for c in new_newick:
        if prev == '#' and c != 'H':
            new_newick_fixed_names += 'H'
        new_newick_fixed_names += c
        prev = c
    return new_newick_fixed_names


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 netrax_output_to_dendroscope.py my_newick.txt")
        exit()
        
    with open(sys.argv[1]) as f:
        newick = f.read()
        f.close()
        print(convert_newick_to_dendroscope(newick))

#!/usr/env/python

import sys
import os

import glob

def extract_taxon_names(paths):
    taxon_names = set()
    for p in paths:
        lines = open(p).readlines()
        name = ""
        seq = ""
        for line in lines:
            if line.startswith(">"):
                taxon_names.add(line[1:-1])
    return taxon_names


def extract_gene_msa(path, taxon_names):
    gene_msa = {}
    for name in taxon_names:
        gene_msa[name] = ""
    lines = open(path).readlines()
    name = ""
    seq = ""
    gene_size = 0
    for line in lines:
        if line.startswith(">"):
            name = line[1:-1]
        else:
            seq = line.replace("\n","")
            gene_msa[name] = seq
            gene_size = len(seq)
    for name in taxon_names:
        if gene_msa[name] == "":
            gene_msa[name] = "-" * gene_size
    return gene_msa, gene_size


paths = [f for f in glob.iglob('OneCopyGenes/*.aln')]

taxon_names = extract_taxon_names(paths)

msa = {}
gene_size = [0 for i in range(len(paths))]
for name in taxon_names:
    msa[name] = []
    
msa_file = open("merged_genes_msa.txt", 'w')
partitions_file = open("merged_genes_partitions.txt", 'w')
start = 1

print(len(paths))
print(len(taxon_names))

print(taxon_names)

species_names = set([t.split("_")[0] + "_" + t.split("_")[1] for t in taxon_names])
print(len(species_names))
print(species_names)
    
for i in range(len(paths)):
    p = paths[i]
    gene_msa, gene_size = extract_gene_msa(p, taxon_names)
    for name in taxon_names:
        msa[name].append(gene_msa[name])
    partitions_file.write("DNA, gene_" + str(i) + "=" + str(start) + "-" + str(start + gene_size - 1) + "\n")
    start = start + gene_size
partitions_file.close()

for name in taxon_names:
    msa_file.write(">" + name + "\n")
    msa_file.write("".join(msa[name]) + "\n")
msa_file.close()
    

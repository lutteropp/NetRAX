#!/usr/env/python

import sys
import os

import glob
import random

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


def extract_species_name(taxon):
    return taxon.split('_')[0] + '_' + taxon.split('_')[1]


def extract_species_seqs(msa, taxon_names, species):
    seqs = []
    for taxon in taxon_names:
        if extract_species_name(taxon) == species:
            seqs.append(msa[taxon])
    return seqs


def majority_consensus_char(species_chars):
    cnt_a = 0
    cnt_c = 0
    cnt_g = 0
    cnt_t = 0
    for char in species_chars:
        if char == 'a' or char == 'A':
            cnt_a += 1
        if char == 'c' or char == 'C':
            cnt_c += 1
        if char == 'g' or char == 'G':
            cnt_g += 1
        if char == 't' or char == 'T':
            cnt_t += 1
    max_cnt = max(cnt_a, cnt_c, cnt_g, cnt_t)
    max_chars = []
    if cnt_a == max_cnt:
        max_chars.append('A')
    if cnt_c == max_cnt:
        max_chars.append('C')
    if cnt_g == max_cnt:
        max_chars.append('G')
    if cnt_t == max_cnt:
        max_chars.append('T')
    if max_cnt == 0:
        return '-'
    return random.choice(max_chars)


def majority_consensus_sequence(species_seqs):
    seq = []
    for idx in range(len(species_seqs[0])):
        chars = [species_seqs[i][idx] for i in range(len(species_seqs))]
        seq.append(majority_consensus_char(chars))
    return "".join(seq)


def build_species_msa(msa, taxon_names, species_names):
    species_msa = {}
    for species in species_names:
        species_seqs = extract_species_seqs(msa, taxon_names, species)
        species_msa[species] = majority_consensus_sequence(species_seqs)
    return species_msa


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
species_msa_file = open("merged_genes_species_msa.txt", 'w')
partitions_file = open("merged_genes_partitions.txt", 'w')
species_partitions_file = open("merged_genes_species_partitions.txt", 'w')
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
    species_partitions_file.write("DNA, gene_" + str(i) + "=" + str(start) + "-" + str(start + gene_size - 1) + "\n")
    start = start + gene_size
partitions_file.close()
species_partitions_file.close()

for name in taxon_names:
    msa_file.write(">" + name + "\n")
    msa_file.write("".join(msa[name]) + "\n")
msa_file.close()
    
species_msa = build_species_msa(msa, taxon_names, species_names)

for name in species_names:
    species_msa_file.write(">" + name + "\n")
    species_msa_file.write("".join(species_msa[name]) + "\n")
species_msa_file.close()


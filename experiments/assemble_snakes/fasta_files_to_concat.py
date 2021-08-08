from Bio import SeqIO
import os


FASTA_FOLDER_PATH = "Lampropelist_NGS_fasta/"

directory = os.fsencode(FASTA_FOLDER_PATH)

seqs = {}
    
snakes_msa = open("snakes_msa.fasta", 'w')
snakes_partitions = open("snakes_partitions.txt", 'w')
    
startpos = 1
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".fasta"):
        lines = open(FASTA_FOLDER_PATH + filename).readlines()
        psize = 0
        taxon = ""
        for line in lines:
            if line.startswith(">"):
                taxon = line[1:].strip()
                if taxon not in seqs:
                    seqs[taxon] = ""
            else:
                seqs[taxon] += line.strip()
                psize = len(line.strip())
        snakes_partitions.write("DNA, " + filename.split(".fasta")[0] + "=" + str(startpos) + "-" + str(startpos+psize-1) + "\n")
        startpos += psize
        
for taxon in seqs:
    snakes_msa.write(">" + taxon + "\n" + seqs[taxon] + "\n")

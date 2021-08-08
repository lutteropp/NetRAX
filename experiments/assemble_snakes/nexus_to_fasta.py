from Bio import SeqIO
import os


NEXUS_FOLDER_PATH = "Lampropelist_NGS/Lampropelist_NGS/"
FASTA_FOLDER_PATH = "Lampropelist_NGS_fasta/"

directory = os.fsencode(NEXUS_FOLDER_PATH)
os.mkdir(FASTA_FOLDER_PATH)
    
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".nex"):
         records = SeqIO.parse(NEXUS_FOLDER_PATH + filename, "nexus")
         count = SeqIO.write(records, FASTA_FOLDER_PATH + filename.split(".nex")[0], "fasta-2line")
         print("Converted %i records" % count)
         continue
     else:
         continue

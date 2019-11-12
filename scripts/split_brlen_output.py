#!/usr/env/python

import sys

filename = sys.argv[1]
infile = open(filename)

mapping = {}
brlens = list()
bID = -1

for line in infile.readlines():
	if line.startswith("Network branch"):
		if (bID != -1):
			mapping[bID] = brlens
		bID = line.split(" ")[2].split(":")[0]
		brlens = list()
	elif line.startswith("  Tree"):
		prob = float(line.split("= ")[1].split(",")[0])
		brlen = float(line.split("= ")[2])
		brlens.append((prob, brlen))
mapping[bID] = brlens

for k in mapping.keys():
	if len(mapping[k]) > 0:
		outfile = open("network_branch_" + str(k) + ".csv", "w")
		outfile.write("tree_prob; brlen\n")
		for e in mapping[k]:
			outfile.write(str(e[0]) + ";" + str(e[1]) + "\n")
		outfile.close()

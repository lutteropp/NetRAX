#!/usr/env/python

import sys
import pandas as pd
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt

infile = open(sys.argv[1])
lines = infile.readlines()
brlens = []
for i in range(len(lines)):
	if i != 0:
		brlens.append(float(lines[i].split(",")[1]))
plt.hist(brlens, bins='auto')
plt.title("Branch-length histogram for " + sys.argv[1].split(".csv")[0])
plt.xlabel('branch length')
plt.ylabel('count')

#plt.show()
plt.savefig(sys.argv[1].split(".csv")[0] + ".png")

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import re

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('-f', '--csvfile', type=str, help='identify a csv file to figure')
parser.add_argument("csvfile", help="identify a csv file to figure")
#parser.add_argument("outfile", help="identify the output filename")
args = parser.parse_args()

csvpath = args.csvfile
#outpath = args.outfile
outpath = csvpath.replace('.csv','.png')
expName = csvpath.split('.')[0]

y = []
count = 1
with open(csvpath,'r') as csvfile:
    next(csvfile)
    read = csv.reader(csvfile, delimiter=' ')
    for row in read:
        y.append(int(row[0]))
        count += 1

x = np.arange(1, count)

ax = plt.axes()
ax.tick_params(direction='in')
ax.grid(linestyle='--')
plt.plot(x,y)
plt.title(expName)
plt.yticks(np.arange(0,16278,3200), ('0%', '20%', '40%', '60%', '80%', '100%'))
plt.xlabel("Total Training Time (ms)")
plt.ylabel("GPU Free Memory")
#plt.show()
plt.savefig(outpath,format='png')


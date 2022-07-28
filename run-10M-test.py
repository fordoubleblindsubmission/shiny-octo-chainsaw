#!/bin/python

import sys
import subprocess
from subprocess import call
from subprocess import Popen, PIPE

# inputs = ['data/zipf_33554432_1.000000', 'data/zipf_33554432_2.000000',
# 'data/zipf_4194304_1.000000',
# 'data/zipf_4194304_2.000000', 'data/unif_1M']

basedir = './data/'
inputs = [
# 'zipf_33554432_0.400000_10000000',
'unif_10M',
'twitter_max'
]

batch_start = 7
batch_end = 2
num_trials = 5

growing_factor = sys.argv[1]
out = sys.argv[2] # open(sys.argv[2], "a")
for f in inputs:
  filename = basedir + f
  print("file: " + f)
  # for i in range(0, batch_end):
  for i in range(batch_start, batch_end, -1):
    batch_size = 10**i
    p = Popen(["./opt_" + growing_factor, "d", str(batch_size), filename, out, str(num_trials)])
    p.wait()

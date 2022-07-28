#!/bin/python

import sys
import subprocess
from subprocess import call
from subprocess import check_call, PIPE

basedir = './data/'

'''
inputs = [
'zipf_67108864_0.400000_100000000',
'zipf_67108864_0.600000_100000000',
'zipf_67108864_0.800000_100000000',
'unif_100M'
]
'''
inputs = [
#'twitter_max',
# 'zipf_4294967296_10000000_0.400000',
'unif_10M'
# 'unif_1B'
# "small_small",
# "small_big",
# "big_big",
# "zipf_67108864_0.400000_10000000"
]

# batch_end = 6


# growing_factor = sys.argv[1]
out = "one_iter" #sys.argv[1] # open(sys.argv[2], "a")
for f in inputs:
  filename = basedir + f
  print("file: " + f)
  batch_start = 6
  batch_end = 5
  num_trials = 5
  # for i in range(0, batch_end):
  for i in range(batch_start, batch_end, -1):
    batch_size = 10**i
    print("batch size: " + str(batch_size))
    # p = check_call(["taskset", "-c", "0-47","./opt", "d", str(batch_size), filename, out, str(num_trials)])
    p = check_call(["taskset", "-c", "0-23","./opt", "d", str(batch_size), filename, out, str(num_trials)])
    p = check_call(["taskset", "-c", "0-15","./opt", "d", str(batch_size), filename, out, str(num_trials)])
    p = check_call(["taskset", "-c", "0-7","./opt", "d", str(batch_size), filename, out, str(num_trials)])
    p = check_call(["taskset", "-c", "0-3","./opt", "d", str(batch_size), filename, out, str(num_trials)])
    p = check_call(["taskset", "-c", "0-1","./opt", "d", str(batch_size), filename, out, str(num_trials)])
    p = check_call(["taskset", "-c", "0","./opt", "d", str(batch_size), filename, out, str(num_trials)])

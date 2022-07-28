import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')

df1 = pd.read_csv('uncompressed_sizes_1.1.csv',
                  names=["1.1"], dtype=np.float32)
# df2 = pd.read_csv('uncompressed_sizes_1.2.csv',
#                   names=["1.2"], dtype=np.float32)
# df3 = pd.read_csv('uncompressed_sizes_1.3.csv',
#                   names=["1.3"], dtype=np.float32)
# df4 = pd.read_csv('uncompressed_sizes_1.4.csv',
#                   names=["1.4"], dtype=np.float32)
df5 = pd.read_csv('uncompressed_sizes_1.5.csv',
                  names=["1.5"], dtype=np.float32)
# df6 = pd.read_csv('uncompressed_sizes_1.6.csv',
#                   names=["1.6"], dtype=np.float32)
# df7 = pd.read_csv('uncompressed_sizes_1.7.csv',
#                   names=["1.7"], dtype=np.float32)
# df8 = pd.read_csv('uncompressed_sizes_1.8.csv',
#                   names=["1.8"], dtype=np.float32)
# df9 = pd.read_csv('uncompressed_sizes_1.9.csv',
#   names = ["1.9"], dtype = np.float32)
df10 = pd.read_csv('uncompressed_sizes_2.0.csv',
                   names=["2.0"], dtype=np.float32)
# df=pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10])
df = pd.concat([df1, df5, df10])
df.plot(ylim=(0, 12), xlim=(100, 10000000))
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

plt.savefig("uncompressed_sizes_growing_factor.png", bbox_inches="tight")
plt.xscale('log')
plt.savefig("uncompressed_sizes_growing_factor_log_x.png",
            bbox_inches="tight")

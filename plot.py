import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
matplotlib.use('Agg')

# df = pd.read_csv(sys.argv[1])
# # df = pd.read_csv('uncompressed_sizes.csv')
# df = df.replace(0, np.nan)
# df.plot(ylim=(0, 16), xlim=(100, len(df.index)))
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# plt.savefig("uncompressed_sizes.png", bbox_inches="tight")
# plt.xscale('log')
# plt.savefig("uncompressed_sizes_log_x.png", bbox_inches="tight")

# df = pd.read_csv(sys.argv[2])
# # df = pd.read_csv('compressed_sizes.csv')
# df = df.replace(0, np.nan)
# df.plot(ylim=(0, 16), xlim=(100, len(df.index)))
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# plt.savefig("compressed_sizes_uint32_t.png", bbox_inches="tight")
# plt.xscale('log')
# plt.savefig("compressed_sizes_log_x_uint32_t.png", bbox_inches="tight")

df_2 = pd.read_csv("uncompressed_sizes_2.0.csv", names=["2.0x"])
df_15 = pd.read_csv("uncompressed_sizes_1.5.csv", names=["1.5x"])
df_11 = pd.read_csv("uncompressed_sizes_1.1.csv", names=["1.1x"])
df = pd.DataFrame()
df["2.0x"] = df_2["2.0x"]
df["1.5x"] = df_15["1.5x"]
df["1.1x"] = df_11["1.1x"]
df = df.replace(0, np.nan)
df.plot(ylim=(0, 24), xlim=(100, len(df.index)))
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

plt.savefig("uncompressed_sizes_uint64_t.png", bbox_inches="tight")
plt.xscale('log')
plt.savefig("uncompressed_sizes_log_x_uint64_t.png", bbox_inches="tight")

df_2 = pd.read_csv("compressed_sizes_2.0.csv", names=["2.0x"])
df_15 = pd.read_csv("compressed_sizes_1.5.csv", names=["1.5x"])
df_11 = pd.read_csv("compressed_sizes_1.1.csv", names=["1.1x"])
df = pd.DataFrame()
df["2.0x"] = df_2["2.0x"]
df["1.5x"] = df_15["1.5x"]
df["1.1x"] = df_11["1.1x"]
df = df.replace(0, np.nan)
df.plot(ylim=(0, 24), xlim=(100, len(df.index)))
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

plt.savefig("compressed_sizes_uint64_t.png", bbox_inches="tight")
plt.xscale('log')
plt.savefig("compressed_sizes_log_x_uint64_t.png", bbox_inches="tight")

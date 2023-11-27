import sys, gzip
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import community as community_louvain

# Read in the data
df = pd.read_csv('cisco+secure+workload+networks+of+computing+hosts\Cisco_22_networks\dir_20_graphs\dir_day1\out1_1.txt.gz', compression='gzip', sep="\t", header=None, names=["src", "dst", "weight"], dtype={"src": str, "dst": str, "weight": str})
df.sort_index(inplace=True)

# Replace final column with computed weight

df["weight"] = df["weight"].map(lambda x: len(x.split(","))) # Count the number of communications between the two hosts

# Convert to networkx graph
G = nx.from_pandas_edgelist(df, "src", "dst", ["weight"], create_using=nx.DiGraph())

# Display the graph
# nx.draw(G, with_labels=True)

# Louvain community detection
partition = nx.community.louvain_communities(G)
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
import sys, gzip
from IPython.display import SVG
import pandas as pd
import numpy as np
import sknetwork as skn
from sknetwork.clustering import Louvain, get_modularity
from sknetwork.linalg import normalize
from sknetwork.utils import get_membership
from sknetwork.visualization import svg_graph, svg_bigraph
import networkx as nx

# Read in the data
df = pd.read_csv('communityDetection/Cisco_22_networks/dir_20_graphs/dir_day1/out1_1.txt.gz', compression='gzip', sep="\t", header=None, names=["graph","src", "dst", "weight"], dtype={"graph": str, "src": str, "dst": str, "weight": str})
df.sort_index(inplace=True)

# Set weight of graph
# df["weight"] = df["weight"].map(lambda x: len(x.split(","))) # Count the number of communications between the two hosts

# Other weight options
df["weight"] = df["weight"].map(lambda x: sum([int(y.split("-")[1]) for y in x.split(",")])) # Count total number of packets sent


# Create adjacency list for graph 1
df_g1 = df.query("graph == 'g1' ")

G = nx.from_pandas_edgelist(df_g1, 'src', 'dst', ['weight'], create_using=nx.DiGraph())
print(G.adjacency_list())

# Create the graph
graph = skn.utils.edgelist2adjacency(df, directed=True, weights="weight", return_edge_list=False)
adjacency = graph.adjacency
position = graph.position

# Create the graph
louvain = Louvain()
labels = louvain.fit_predict(adjacency)

image = svg_graph(adjacency, position, labels=labels)
SVG(image)
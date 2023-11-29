import pandas as pd
import numpy as np
import networkx as nx
from karateclub.node_embedding.neighbourhood.deepwalk import DeepWalk
import matplotlib.pyplot as plt


# Read in the data from g1_1.csv
df = pd.read_csv('communityDetection\graphs\g1_1.csv', sep="\t", header=None, names=["graph","src", "dst", "weight"], dtype={"graph": str, "src": int, "dst": int, "weight": str})
df.sort_index(inplace=True)

# update index of source and destination nodes
df["src"] = df["src"].map(lambda x: x - 1)
df["dst"] = df["dst"].map(lambda x: x - 1) 

# Set weights of graph
df["weight"] = df["weight"].map(lambda x: len(x.split(","))) # Count the number of communications between the two hosts

# Create a graph from the dataframe
G = nx.from_pandas_edgelist(df, "src", "dst", edge_attr="weight", create_using=nx.Graph())
#nx.draw(G, with_labels=True)
#plt.show()
print(G.nodes)


# Create a DeepWalk model
deepwalk = DeepWalk(dimensions=2)
deepwalk.fit(G)

embedding = deepwalk.get_embedding()
print(embedding)

plt.scatter(embedding[:,0], embedding[:,1])
plt.show()
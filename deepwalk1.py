import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from karateclub import DeepWalk

# import dataset
df = pd.read_csv('comunityDetPrj\graphs\g1_1.csv', sep="\t", header=None, names=["graph","src", "dst", "weight"], dtype={"graph": str, "src": int, "dst": int, "weight": str})
df.sort_index(inplace=True)
df.head()

# create Graph
G = nx.from_pandas_edgelist(df, "node_1", "node_2", create_using=nx.Graph())
print(len(G))

# train model and generate embedding
model = DeepWalk(walk_length=100, dimensions=64, window_size=5)
model.fit(G)
embedding = model.get_embedding()

# print Embedding shape
print(embedding.shape)
# take first 100 nodes
nodes = list(range(100))


# plot nodes graph
def plot_nodes(node_no):
    X = embedding[node_no]

    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(X)

    plt.figure(figsize=(15, 10))
    plt.scatter(pca_out[:, 0], pca_out[:, 1])
    for i, node in enumerate(node_no):
        plt.annotate(node, (pca_out[i, 0], pca_out[i, 1]))
    plt.xlabel('Label_1')
    plt.ylabel('label_2')
    plt.show()


plot_nodes(nodes)

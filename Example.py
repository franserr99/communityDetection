import pandas as pd
import numpy as np
import networkx as nx
#from karateclub.node_embedding.neighbourhood.deepwalk import DeepWalk
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# from networkx.algorithms import community

# Read in the data from g1_1.csv
df = pd.read_csv('graphs/g1_1.csv', sep="\t", header=None, names=["graph","src", "dst", "weight"], dtype={"graph": str, "src": int, "dst": int, "weight": str})
df.sort_index(inplace=True)

# Set weights of graph
df["weight"] = df["weight"].map(lambda x: len(x.split(","))) # Count the number of communications between the two hosts

# TODO: Check other weight options (e.g.  total number of packets, weighted sum with packet size, etc.)

# Create a graph from the dataframe
G = nx.from_pandas_edgelist(df, "src", "dst", edge_attr="weight", create_using=nx.Graph())
#nx.draw(G, with_labels=True)
#plt.show()

# Supposed True Communities
# y = community.louvain_communities(G) # True Labels from Louvain Algorithm

 # Create a DeepWalk model
"""
deepwalk = DeepWalk(dimensions=2)
deepwalk.fit(G)

embedding = deepwalk.get_embedding()

# plt.scatter(embedding[:,0], embedding[:,1])
"""
best_model = None
best_accuracy = 0
best_cluster_size = 0
best_window_size = 0

for window_size in [5]:
    # Create a StellarGraph model
    walk_length = 100
    G = StellarGraph.from_networkx(G, node_features=None, edge_weight_attr="weight")
    rw = BiasedRandomWalk(G)
    walks = rw.run(nodes=list(G.nodes()), length=walk_length, n=10, p=0.3, q=2.0, seed=42)
    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, vector_size=128, window=window_size, min_count=0, sg=1, workers=2, epochs=5)
    node_ids = model.wv.index_to_key
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    embedding = TSNE(n_components=2).fit_transform(node_embeddings)


    for model in ["mediods", "kmeans","spectral"]:
        for cluster_size in [2, 3, 4, 5, 6, 7, 8, 9, 10]:

            if model == "mediods":
                # Create a KMedoids model
                kmedoids = KMedoids(n_clusters=cluster_size, random_state=0).fit(embedding)
                labels = kmedoids.labels_

            elif model == "kmeans":
                # Create a KMeans model
                kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(embedding)
                labels = kmeans.labels_
            elif model == "spectral":
                # Create a SpectralClustering model
                spectral = SpectralClustering(n_clusters=cluster_size, random_state=0).fit(embedding)
                labels = spectral.labels_

            # Compute Sihouette Score
            score = silhouette_score(embedding, labels, metric="euclidean")

            if score > best_accuracy:
                best_accuracy = score
                best_model = model
                best_cluster_size = cluster_size
                best_window_size = window_size
                # display best model and cluster size
                print("Best Model: ", best_model)
                print("Best Cluster Size: ", best_cluster_size)
                print("Best Accuracy: ", best_accuracy)
                print("Best Window Size: ", best_window_size)

# display best model and cluster size
print("Best Model: ", best_model)
print("Best Cluster Size: ", best_cluster_size)
print("Best Accuracy: ", best_accuracy)
print("Best Window Size: ", best_window_size)

# TODO: Check other clustering algorithms (e.g. Spectral Clustering, K-means, etc.)

# Compute accuracy (I think...?)
# accuracy = accuracy_score([y[node] for node in range(len(y))], labels, normalize=True)


""" # Display Clustering (Uncecessary)
unique_labels = set(labels)
colors = [
    plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
]
for k, col in zip(unique_labels, colors):
    class_member_mask = labels == k

    xy = embedding[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.plot(
    kmedoids.cluster_centers_[:, 0],
    kmedoids.cluster_centers_[:, 1],
    "o",
    markerfacecolor="cyan",
    markeredgecolor="k",
    markersize=6,
)

plt.title("KMedoids clustering. Medoids are represented in cyan.")
plt.show()  """
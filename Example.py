import pandas as pd
# import numpy as np
import networkx as nx
from karateclub.node_embedding.neighbourhood.deepwalk import DeepWalk
# import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
# from sklearn import preprocessing
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarDiGraph, StellarGraph
from gensim.models import Word2Vec
# import community
from typing import TypedDict, Dict, List, Union
from sklearn.metrics import silhouette_score
from enum import Enum


class ClusterMembership(TypedDict):
    day: Dict[int, Dict[int, List[int]]]


class WeightMethod(Enum):
    NUM_OF_COMM = "num_of_comm"
    # TODO :add enums for ryans work +
    #       their implementation in code or selection from file w it already computed


class EmbeddingMethod(Enum):
    WEIGHTED_RANDOM = "weighted_random"
    DEEP_WALK = "deep_walk"


class ClusteringMethod(Enum):
    KMEDOIDS = "kmedoids"
    KMEANS = "kmeans"
    SPECTRAL = "spectral"
    DBSCAN = "dbscan"
    AGGCLUSTERING = "agglomerative_clustering"


class ModelParameters(TypedDict):
    walk_length: int
    num_of_walks: int
    vec_size: int
    num_of_clusters: int
    embedding: EmbeddingMethod
    weight: WeightMethod
    clustering: ClusteringMethod


def main():
    # switch this to iterate all graphs for now keep to 1
    for i in range(1):
        cluster_membership: ClusterMembership = {}
        for j in range(1):
            # hyper-paramters (tuning on a per data set basis (best parameters for each graph))
            # we can use the ground truth model as a validation set for some selection of parameters we choose
            # and find some way to get a general set of "best parameters" for the domain/data set
            walk_lengths = [100]
            num_of_random_walks = []
            # starting with larger intervals we can narrow down (idk the scale of # of clusters)
            num_of_clusters = [4, 8, 16, 20]
            # we need to tune the important params in each of these
            embedding_methods = [
                EmbeddingMethod.WEIGHTED_RANDOM, EmbeddingMethod.DEEP_WALK]
            weight_methods = [WeightMethod.NUM_OF_COMM]
            clustering_methods = [ClusteringMethod.KMEANS,
                                  ClusteringMethod.KMEDOIDS,
                                  ClusteringMethod.SPECTRAL,
                                  ClusteringMethod.DBSCAN]
            # -1 -> 1 range for silhoutte score
            max_score = -1
            for walk_length in walk_lengths:
                for n_walk in num_of_random_walks:
                    for n_clusters in num_of_clusters:
                        paramters = ModelParameters(walk_length=walk_length,
                                                    num_of_clusters=n_clusters,
                                                    num_of_random_walks=n_walk)

                        create_clustering(paramters, i, j, cluster_membership)


def perform_kmediods(parameters: ModelParameters, embedding):
    kmedoids = KMedoids(
        n_clusters=parameters['num_of_clusters'],
        random_state=0).fit(embedding)
    labels = kmedoids.labels_
    sse = kmedoids.inertia_
    score = silhouette_score(embedding, labels)
    return kmedoids, labels, sse, score


def perform_kmeans(parameters: ModelParameters, embedding):
    kmeans = KMeans(n_clusters=parameters['num_of_clusters'],
                    random_state=0).fit(embedding)
    labels = kmeans.labels_
    sse = kmeans.inertia_
    score = silhouette_score(embedding, labels)
    return kmeans, labels, sse, score


def create_clustering(parameters: ModelParameters, i: int, j: int,
                      cluster_membership: ClusterMembership):
    cluster_membership[j] = {}
    df = get_df(i, j, WeightMethod.NUM_OF_COMM)
    if parameters['embedding_method'] == EmbeddingMethod.WEIGHTED_RANDOM:
        best_score = -1
        best_p = 0
        best_q = 0
        best_clustering_method = None
        best_clustering_parameters = {}
        G = get_stellar_graph(df)
        Ps = [0.5]
        Qs = [2.0]
        for p in Ps:
            for q in Qs:
                model, embedding = get_embedding_info(
                    G, parameters['walk_length'], p, q, parameters['vec_size'])
                # create functions for the repetitive stuff
                if parameters['clustering'] == ClusteringMethod.KMEDOIDS:
                    kmedoids, labels, sse, score = perform_kmediods(
                        parameters, embedding)
                    # not sure if i wannna return the score?
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_q = q
                        best_clustering_method = ClusteringMethod.KMEDOIDS
                    return score
                elif parameters['clustering'] == ClusteringMethod.KMEANS:
                    kmeans, labels, sse, score = perform_kmeans(parameters,
                                                                embedding)
                    # cluster_centers = kmeans.cluster_centers_
                    score = silhouette_score(embedding, labels)
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_q = q
                        best_clustering_method = ClusteringMethod.KMEANS
                    return score
                elif parameters['clustering'] == ClusteringMethod.SPECTRAL:
                    # it has an affinity field we can change
                    # 'nearest_neighbors' or 'rbf' or an affinity matrix
                    # (if we have time?)
                    spectral_clustering = SpectralClustering(
                        n_clusters=parameters['num_of_clusters'],
                        random_state=0).fit(embedding)
                    labels = spectral_clustering.labels_
                    # does not have an inertia field (not applicable?)
                    score = silhouette_score(embedding, labels)
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_q = q
                        best_clustering_method = ClusteringMethod.SPECTRAL
                    return score
                elif (parameters['clustering'] == ClusteringMethod.DBSCAN):
                    eps = [0.3, 0.5, 0.7]
                    min_samples = [5, 10, 20, 100]
                    best_eps = 0
                    best_min_sample = 0
                    for ep in eps:
                        for min_sample in min_samples:
                            dbscan = DBSCAN(
                                eps=ep, min_samples=min_sample).fit(embedding)
                            labels = dbscan.labels_
                            score = silhouette_score(embedding, labels)
                            if score > best_score:
                                best_score = score
                                best_p = p
                                best_q = q
                                best_eps = ep
                                best_min_sample = min_sample
                                best_clustering_method = ClusteringMethod.DBSCAN
                elif (parameters['clustering'] == ClusteringMethod.AGGCLUSTERING):
                    linkages = ['ward', 'average']
                    best_linkage = ''
                    for linkage in linkages:
                        agg_cluster = AgglomerativeClustering(
                            n_clusters=parameters['num_of_clusters'],
                            linkage=linkage).fit(embedding)
                        labels = agg_cluster.labels_
                        score = silhouette_score()
                        if score > best_score:
                            best_score = score
                            best_p = p
                            best_q = q
                            best_linkage = linkage
                            best_clustering_method = ClusteringMethod.AGGCLUSTERING
                            return score
        print("the best p and q we found was:", best_p, best_q)
    elif parameters['embedding_method'] == EmbeddingMethod.DEEP_WALK:
        deepwalk = DeepWalk(dimensions=2)
        deepwalk.fit(G)
        embedding = deepwalk.get_embedding()
        # plt.scatter(embedding[:,0], embedding[:,1])

    # TODO: Check other clustering algorithms (e.g. Spectral Clustering, K-means, etc.)
    # Create a KMedoids model


def get_embedding_info(G: Union[StellarGraph, StellarDiGraph], walk_length, p, q, vec_size):
    rw = BiasedRandomWalk(G)
    walks = rw.run(nodes=list(G.nodes()), length=walk_length,
                   n=10, p=p, q=q, seed=42)
    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, vector_size=vec_size, window=5,
                     min_count=0, sg=1, workers=2, epochs=1)
    node_ids = model.wv.index_to_key
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    embedding = TSNE(n_components=2).fit_transform(
        node_embeddings)
    # when i refactored i only returned model and embedding because you can access the rest thru them
    return model, embedding


def get_df(graph: int, day: int, method: WeightMethod):
    df = pd.read_csv(f'graphs/g{graph+1}_{day+1}.csv', sep="\t", header=None, names=[
        "graph", "src", "dst", "weight"], dtype={"graph": str, "src": int, "dst": int, "weight": str})
    df.sort_index(inplace=True)
    # TODO: Check other weight options (e.g.  total number of packets, weighted sum with packet size, etc.)
    if method == WeightMethod.NUM_OF_COMM:
        # Set weights of graph
        # Count the number of communications between the two hosts
        df["weight"] = df["weight"].map(lambda x: len(x.split(",")))
    return df


def get_stellar_graph(df: pd.DataFrame):
    # Create a graph from the dataframe
    G = nx.from_pandas_edgelist(
        df, "src", "dst", edge_attr="weight", create_using=nx.Graph())
    # nx.draw(G, with_labels=True)
    # plt.show()
    G = StellarGraph.from_networkx(
        G, node_features=None, edge_weight_attr="weight")
    return G


if __name__ == "__main__":
    main()

""" unique_labels = set(labels)
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
plt.show() """

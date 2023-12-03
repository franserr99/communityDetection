import pandas as pd
import numpy as np
import networkx as nx
from karateclub.node_embedding.neighbourhood.deepwalk import DeepWalk
# import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, \
    AgglomerativeClustering
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
            # hyper-paramters (tuning on a per data set basis
            # (best parameters for each graph))
            # we can use the ground truth model as a validation set for
            # some selection of parameters we choose
            # and find some way to get a general set of "best parameters" for the domain/data set
            walk_lengths = [100]
            num_of_random_walks = [10]
            vector_sizes = [128]
            # starting with larger intervals we can narrow down (idk the scale of # of clusters)
            num_of_clusters = [4, 8, 16, 20]
            # we need to tune the important params in each of these
            # onky do weighted random for now
            embedding_methods = [
                EmbeddingMethod.WEIGHTED_RANDOM]
            weight_methods = [WeightMethod.NUM_OF_COMM]
            clustering_methods = [ClusteringMethod.KMEANS,
                                  ClusteringMethod.KMEDOIDS,
                                  ClusteringMethod.SPECTRAL,
                                  ClusteringMethod.DBSCAN]
            # -1 -> 1 range for silhoutte score
            max_score = -1
            best_shared_param = {}
            best_addtl_param = {}
            for walk_length in walk_lengths:
                for n_walk in num_of_random_walks:
                    for n_clusters in num_of_clusters:
                        for embedding_method in embedding_methods:
                            for clustering_method in clustering_methods:
                                for vect_size in vector_sizes:
                                    for weight_method in weight_methods:
                                        parameters = ModelParameters(
                                            walk_length=walk_length,
                                            num_of_clusters=n_clusters,
                                            num_of_walks=n_walk,
                                            vec_size=vect_size,
                                            embedding=embedding_method,
                                            clustering=clustering_method,
                                            weight=weight_method
                                        )
                                        all_addtl_params = create_clustering(
                                            parameters, i, j,
                                            cluster_membership)
                                        score = all_addtl_params['score']
                                        if score > max_score:
                                            max_score = score
                                            best_shared_param = parameters
                                            best_addtl_param = all_addtl_params
                                            print(
                                                "The best found so far we found was:")
                                            print(best_shared_param)
                                            print(best_addtl_param)
            print("The best clustering we found was:")
            print(best_shared_param)
            print(best_addtl_param)


def perform_kmediods(parameters: ModelParameters, embedding):
    kmedoids = KMedoids(
        n_clusters=parameters['num_of_clusters'],
        random_state=0).fit(embedding)
    labels = kmedoids.labels_
    sse = kmedoids.inertia_
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        # more than one cluster exists, calculate silhouette score
        score = silhouette_score(embedding, labels)
    else:
        # only one cluster
        score = -1
    return kmedoids, labels, sse, score


def perform_kmeans(parameters: ModelParameters, embedding):
    kmeans = KMeans(n_clusters=parameters['num_of_clusters'],
                    random_state=0).fit(embedding)
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        # more than one cluster exists, calculate silhouette score
        score = silhouette_score(embedding, labels)
    else:
        # only one cluster
        score = -1
    sse = kmeans.inertia_
    return kmeans, labels, sse, score


def perform_spectral(parameters: ModelParameters, embedding):
    spectral_clustering = SpectralClustering(
        n_clusters=parameters['num_of_clusters'],
        random_state=0).fit(embedding)
    labels = spectral_clustering.labels_
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        # more than one cluster exists, calculate silhouette score
        score = silhouette_score(embedding, labels)
    else:
        # only one cluster
        score = -1
    # does not have an inertia field (not applicable?)
    return spectral_clustering, labels, score


def perform_db_scan(parameters: ModelParameters, embedding):
    eps = [0.1, 0.2, 0.4]
    min_samples = [5, 10, 20]
    best_eps = 0
    best_min_sample = 0
    # find the best db clustering to compare against the
    # rest for the set of param handed
    best_db_score = 0
    best_db_scan = None
    for ep in eps:
        for min_sample in min_samples:
            dbscan = DBSCAN(
                eps=ep, min_samples=min_sample).fit(embedding)
            labels = dbscan.labels_
            unique_labels = np.unique(labels)

            if len(unique_labels) > 1:
                # more than one cluster exists, calculate silhouette score
                score = silhouette_score(embedding, labels)
            else:
                # only one cluster
                score = -1
            if score > best_db_score:
                best_db_score = score
                # best_p = p
                # best_q = q
                best_eps = ep
                best_min_sample = min_sample
                best_db_scan = dbscan
    return best_db_scan, best_db_score, best_eps, best_min_sample


def create_clustering(parameters: ModelParameters, i: int, j: int,
                      cluster_membership: ClusterMembership):

    cluster_membership[j] = {}
    df = get_df(i, j, parameters['weight'])
    best_embedding_method = None
    all_addtl_params = {}
    best_score = -1
    if parameters['embedding'] == EmbeddingMethod.WEIGHTED_RANDOM:
        best_embedding_method = EmbeddingMethod.WEIGHTED_RANDOM

        best_p = 0
        best_q = 0
        best_clustering_method = None
        # best_clustering_parameters is mostly for agg clustering and db scan
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
                elif parameters['clustering'] == ClusteringMethod.KMEANS:
                    kmeans, labels, sse, score = perform_kmeans(parameters,
                                                                embedding)
                    # cluster_centers = kmeans.cluster_centers_
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_q = q
                        best_clustering_method = ClusteringMethod.KMEANS
                elif parameters['clustering'] == ClusteringMethod.SPECTRAL:
                    # it has an affinity field we can change
                    # 'nearest_neighbors' or 'rbf' or an affinity matrix
                    # (if we have time?)
                    spectral_clustering, labels, score = perform_spectral(
                        parameters, embedding)
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_q = q
                        best_clustering_method = ClusteringMethod.SPECTRAL
                elif (parameters['clustering'] == ClusteringMethod.DBSCAN):
                    best_db_scan, best_db_score, best_eps, best_min_sample = \
                        perform_db_scan(parameters, embedding)
                    if best_db_score > best_score:
                        best_score = score
                        best_p = p
                        best_q = q
                        best_clustering_method = ClusteringMethod.DBSCAN
                        best_clustering_parameters = {
                            'eps': best_eps, 'min_sample': best_min_sample}
                elif (parameters['clustering'] ==
                      ClusteringMethod.AGGCLUSTERING):
                    # linkages = ['ward', 'average']
                    # best_linkage = ''
                    # for linkage in linkages:
                    agg_cluster = AgglomerativeClustering(
                        n_clusters=parameters['num_of_clusters']
                    ).fit(embedding)
                    labels = agg_cluster.labels_
                    unique_labels = np.unique(labels)
                    if len(unique_labels) > 1:
                        # more than one cluster exists, calculate silhouette score
                        score = silhouette_score(embedding, labels)
                    else:
                        # only one cluster
                        score = -1  
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_q = q
                        # best_linkage = linkage
                        best_clustering_method = ClusteringMethod.AGGCLUSTERING
        print("the best p and q we found was:", best_p, best_q)
        all_addtl_params.update(best_clustering_parameters)
        all_addtl_params.update({'p': best_p, 'q': best_q})
        all_addtl_params.update({'clustering_method': best_clustering_method})
        all_addtl_params.update({'embedding_method': best_embedding_method})
        all_addtl_params.update({'score': best_score})

    elif parameters['embedding'] == EmbeddingMethod.DEEP_WALK:
        deepwalk = DeepWalk(dimensions=2)
        deepwalk.fit(G)
        embedding = deepwalk.get_embedding()

        # plt.scatter(embedding[:,0], embedding[:,1])

        # that big if elif elif in the weighted one shows up here
        # debug for the first part
        #  then try to refactor it so that it can be a function
        # it would make this part cleaner

    # TODO: Check other clustering algorithms
    # (e.g. Spectral Clustering, K-means, etc.)
    # Create a KMedoids model
    return all_addtl_params


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
    # when i refactored i only returned model and embedding because
    # you can access the rest thru them
    return model, embedding


def get_df(graph: int, day: int, method: WeightMethod):
    df = pd.read_csv(f'graphs/g{graph+1}_{day+1}.csv',
                     sep="\t", header=None,
                     names=["graph", "src", "dst", "weight"],
                     dtype={"graph": str, "src": int,
                            "dst": int, "weight": str})
    df.sort_index(inplace=True)
    # TODO: Check other weight options (e.g.  total number of packets,
    #  weighted sum with packet size, etc.)
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

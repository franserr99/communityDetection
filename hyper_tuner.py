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
import os

"""
from biweight import avgWeights, medianWeight
from logWeight import np, edge
from connectionWeight import edgeConnections
"""

class ClusterMembership(TypedDict):
    day: Dict[int, Dict[int, List[int]]]


class WeightMethod(Enum):
    NUM_OF_COMM = "num_of_comm"
    # TODO :add enums for ryans work +
    #       their implementation in code or selection from file w it already computed
    BIWEIGHT = "biweight"
    LOG_WEIGHT = "log_weight"
    CONNECTION_WEIGHT = "connection_weight"

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
    directory = "graphs"


    files = ["graphs/g" +str(i) + "_" + str(j) + ".csv" for i in [3,4,5,6,7,8,9] for j in range(1,5)]

    for file in files:
    #for file in os.listdir(directory):
        counter = 0
        #file = os.path.join(directory, file)
        print(file)
        cluster_membership: ClusterMembership = {}            # hyper-paramters (tuning on a per data set basis
        # (best parameters for each graph))
        # we can use the ground truth model as a validation set for
        # some selection of parameters we choose
        # and find some way to get a general set of "best parameters" for the domain/data set
        walk_lengths = [100]
        num_of_random_walks = [10]
        vector_sizes = [128]
        # starting with larger intervals we can narrow down (idk the scale of # of clusters)
        num_of_clusters = [4]
        # we need to tune the important params in each of these
        # onky do weighted random for now
        embedding_methods = [EmbeddingMethod.DEEP_WALK]
        weight_methods = [WeightMethod.NUM_OF_COMM]

        clustering_methods = [ClusteringMethod.DBSCAN]
        # -1 -> 1 range for silhoutte score
        max_score = -1
        best_shared_param = {}
        best_addtl_param = {}

        # for each graph
        for i in range(1):
            for weight_method in weight_methods:
                # i think moving this upwards should make it faster?
                df = get_df(file, WeightMethod.NUM_OF_COMM)
                print(df.head())
                for walk_length in walk_lengths:
                    for n_walk in num_of_random_walks:
                        for n_clusters in num_of_clusters:
                            for embedding_method in embedding_methods:
                                for clustering_method in clustering_methods:
                                    for vect_size in vector_sizes:

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
                                            parameters, counter,
                                            cluster_membership, df)
                                        score = all_addtl_params['score']
                                        if score > max_score:
                                            max_score = score
                                            best_shared_param = parameters
                                            best_addtl_param = all_addtl_params

            # add to results
            graph = file.split('/')[-1]
            graph = graph.split('.')[0]
            graph,day = graph.split('_')[0],graph.split('_')[1]
            
            # output results to results.txt
            output = open("Experiment.txt", "a")
            output.write("Graph: " + graph + "\n")
            output.write("Day: " + day + "\n")
            output.write("Walk Length: " + str(best_shared_param['walk_length']) + "\n")
            output.write("Number of Walks: " + str(best_shared_param['num_of_walks']) + "\n")
            output.write("Vector Size: " + str(best_shared_param['vec_size']) + "\n")
            output.write("Number of Clusters: " + str(best_shared_param['num_of_clusters']) + "\n")
            output.write("Embedding Method: " + str(best_shared_param['embedding']) + "\n")
            if best_addtl_param['embedding_method'] == EmbeddingMethod.WEIGHTED_RANDOM:
                output.write("P: " + str(best_addtl_param['p']) + "\n")
                output.write("Q: " + str(best_addtl_param['q']) + "\n")
            output.write("Weight Method: " + str(best_shared_param['weight']) + "\n")
            output.write("Clustering Method: " + str(best_shared_param['clustering']) + "\n")
            output.write("Score: " + str(best_addtl_param['score']) + "\n")
            output.write("\n")
            output.close()



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
    print(unique_labels)
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


def create_clustering(parameters: ModelParameters, j: int,
                      cluster_membership: ClusterMembership, df: pd.DataFrame):

    cluster_membership[j] = {}

    best_embedding_method = None
    all_addtl_params = {}
    best_score = -1
    best_clustering_method = None

    # best_clustering_parameters is mostly for agg clustering and db scan
    best_clustering_parameters = {}

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
                        best_score = best_db_score
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
        # print("the best p and q we found was:", best_p, best_q)
        all_addtl_params.update(best_clustering_parameters)
        all_addtl_params.update({'p': best_p, 'q': best_q})
        all_addtl_params.update({'clustering_method': best_clustering_method})
        all_addtl_params.update({'embedding_method': best_embedding_method})
        all_addtl_params.update({'score': best_score})

    elif parameters['embedding'] == EmbeddingMethod.DEEP_WALK:

        # Get networkx graph from dataframe
        G = G = nx.from_pandas_edgelist(df, "src", "dst", edge_attr="weight", create_using=nx.Graph())

        deepwalk = DeepWalk(dimensions=2)
        deepwalk.fit(G)
        embedding = deepwalk.get_embedding()

        if parameters['clustering'] == ClusteringMethod.KMEDOIDS:
            kmedoids, labels, sse, score = perform_kmediods(
                parameters, embedding)
            # not sure if i wannna return the score?
            if score > best_score:
                best_score = score
                best_clustering_method = ClusteringMethod.KMEDOIDS

        elif parameters['clustering'] == ClusteringMethod.KMEANS:
            kmeans, labels, sse, score = perform_kmeans(parameters,
                                                        embedding)
            # cluster_centers = kmeans.cluster_centers_
            if score > best_score:
                best_score = score
                best_clustering_method = ClusteringMethod.KMEANS

        elif parameters['clustering'] == ClusteringMethod.SPECTRAL:
            # it has an affinity field we can change
            # 'nearest_neighbors' or 'rbf' or an affinity matrix
            # (if we have time?)
            spectral_clustering, labels, score = perform_spectral(
                parameters, embedding)
            if score > best_score:
                best_score = score
                best_clustering_method = ClusteringMethod.SPECTRAL

        elif (parameters['clustering'] == ClusteringMethod.DBSCAN):
            best_db_scan, best_db_score, best_eps, best_min_sample = \
                perform_db_scan(parameters, embedding)
            if best_db_score > best_score:
                best_score = best_db_score
                best_clustering_method = ClusteringMethod.DBSCAN
                best_clustering_parameters = {
                    'eps': best_eps, 'min_sample': best_min_sample}

        elif (parameters['clustering'] == ClusteringMethod.AGGCLUSTERING):
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
                # best_linkage = linkage
                best_clustering_method = ClusteringMethod.AGGCLUSTERING

    all_addtl_params.update(best_clustering_parameters)
    all_addtl_params.update({'clustering_method': best_clustering_method})
    all_addtl_params.update({'embedding_method': best_embedding_method})
    all_addtl_params.update({'score': best_score})

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
    embedding = TSNE(n_components=2, perplexity=5).fit_transform(
        node_embeddings)
    # when i refactored i only returned model and embedding because
    # you can access the rest thru them
    return model, embedding

def count_communications(x: str) -> int:
    return len(x.split(","))

# Define separate functions for each weight method
def calculate_weight_num_of_comm(x: str) -> int:
    return count_communications(x)

def calculate_biweight_weight(edge: tuple) -> float:
    weight = avgWeights[edge]
    biweight = (weight - medianWeight) / (9 * medianWeight)
    return biweight

def calculate_log_weight_weight(edge: tuple) -> float:
    count = edge[edge]
    log_weight = 0.5 * np.log(count + 1)
    return log_weight

def calculate_connection_weight_weight(edge: tuple) -> int:
    return edgeConnections[edge]

def get_df(file: str, method: WeightMethod):
    df = pd.read_csv(file,
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
    
    elif method == WeightMethod.BIWEIGHT:
        df["weight"] = df.apply(lambda row: calculate_biweight_weight(
            (row["graph"], row["src"], row["dst"])), axis=1)
    elif method == WeightMethod.LOG_WEIGHT:
        df["weight"] = df.apply(lambda row: calculate_log_weight_weight(
            (row["graph"], row["src"], row["dst"])), axis=1)
    elif method == WeightMethod.CONNECTION_WEIGHT:
        df["weight"] = df.apply(lambda row: calculate_connection_weight_weight(
            (row["graph"], row["src"], row["dst"])), axis=1)

#            df["weight"] = df["weight"].map(calculate_weight_num_of_comm)
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

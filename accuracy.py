import pandas as pd
import networkx as nx
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
from typing import TypedDict, Dict, Set, Union
import os
from stellargraph import StellarDiGraph


class ClusterMembership(TypedDict):
    day: Dict[int, Dict[int, Set[int]]]
# path to the ground truth


def getAggregatedDF():
    dir_path = 'Cisco_22_networks/dir_g21_small_workload_with_gt/dir_includes_packets_and_other_nodes'
    dfs = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.gz'):
            filepath = os.path.join(dir_path, filename)
            df = pd.read_csv(filepath, compression='gzip', sep="\t", header=None,
                             names=["graph", "src", "dst", "weight"],
                             dtype={"graph": str, "src": int, "dst": int, "weight": str})
            dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)
    df = final_df
    df.sort_index(inplace=True)
    # Set weights of graph
    # Count the number of communications between the two hosts
    df["weight"] = df["weight"].map(lambda x: len(x.split(",")))
    # aggregate the values
    aggregated_df = df.groupby(['src', 'dst'])['weight'].sum().reset_index()
    print(aggregated_df)
    return aggregated_df


def getStellarGraph(df: pd.DataFrame):
    G = nx.from_pandas_edgelist(
        df, "src", "dst", edge_attr="weight", create_using=nx.Graph())
    # Create a StellarGraph model

    G = StellarGraph.from_networkx(
        G, node_features=None, edge_weight_attr="weight")
    return G


def getNodeEmbeddingInfo(graph: Union[StellarGraph, StellarDiGraph]):
    walk_length = 100
    rw = BiasedRandomWalk(graph)
    walks = rw.run(nodes=list(graph.nodes()), length=walk_length,
                   n=10, p=0.5, q=2.0, seed=42)
    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, vector_size=128, window=5,
                     min_count=0, sg=1, workers=2, epochs=1)
    node_ids = model.wv.index_to_key
    node_embeddings = (
        model.wv.vectors
    )
    return node_ids, node_embeddings, model


def getGroundTruthClusters():
    file_path = 'Cisco_22_networks/dir_g21_small_workload_with_gt/groupings.gt.txt'
    ground_truth_clusters = []
    with open(file_path, 'r') as file:
        for line in file:
            cluster = set(map(int, line.strip().split(',')))
            ground_truth_clusters.append(cluster)
    return ground_truth_clusters


def main():
    df = getAggregatedDF()
    # Create a graph from the dataframe
    G = getStellarGraph(df)
    node_ids, node_embeddings, model = getNodeEmbeddingInfo(G)
    # numpy.ndarray of size number of nodes times embeddings dimensionality
    embedding = TSNE(n_components=2).fit_transform(node_embeddings)
    # Create a KMedoids model
    kmedoids = KMedoids(n_clusters=4, random_state=0).fit(embedding)
    model_labels = kmedoids.labels_
    centroids = kmedoids.cluster_centers_


    ground_truth_clusters = getGroundTruthClusters()
    print(ground_truth_clusters)
    # unpack the array args into the union
    ground_truth_set = set().union(*ground_truth_clusters)
    # node_ids is a list of strings (i assume they use the client and server ID but convert to str)
    common_nodes = set(map(int, node_ids)) & ground_truth_set
    # init sets for each cluster
    filtered_model_clusters = [set() for i in range(kmedoids.n_clusters)]
    # get nodeID, filter to only the ones in ground truth
    for index, label in enumerate(model_labels):
        nodeID = int(node_ids[index])
        if nodeID in common_nodes:
            filtered_model_clusters[label].add(nodeID)
    print(common_nodes)
    # assert ground_truth_set == common_nodes
    # i think some nodes in ground truth arent in the common nodes
    filtered_ground_truth_clusters = [cluster & common_nodes for cluster in ground_truth_clusters]
    print(filtered_ground_truth_clusters)


if __name__ == "__main__":
    main()

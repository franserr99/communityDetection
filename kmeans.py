import pandas as pd
import numpy as np


# Read in the data from g1_1.csv
df = pd.read_csv('graphs/g1_1.csv', sep="\t", header=None, names=["graph","src", "dst", "weight"], dtype={"graph": str, "src": int, "dst": int, "weight": str})

# Set weights of graph
features = df["weight"].map(lambda x: len(x.split(",")))

# copy some rows data into a new dataframe to use in clustering
data = df[features].copy()

#intialize random k centroids
def random_centroids(data, k):
    centroids =[]
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

# generate centroids of k=5 clusters
centroids = random_centroids(data, 5)   #Note: this should display a table where each coulmn is a centroids, and rows are the features


def get_labels(data, centroids):
    # calculate the Euclidean distance from each datapoint to each centroid
    distances = centroids.apply(np.sqrt(((data - centroids.iloc[:,0]) ** 2 ).sum(axis=1))) # [for each centroid] subtract the value of the centroid - value of each data point (for all features), square each differenc, and finally add all difference together to apply sqrt for the result
    # find the index of the closest cluster to a datapoint (e.g 0, 1, 2, 3, 4)
    return distances.idxmin(axis=1)

labels = get_labels(data, centroids)

# (Optional) to check the number of datapoints that are assigned to each cluster...
#labels.value_counts()

# find the geometric mean to get the new centroids
def new_centroids(data, labels, k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T  #Note: this should display a table where each coulmn is a centroids, and rows are the features

# Graph clusters into 2D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2) # we want to get data in 2 cloumns e.g. 2D
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()

max_iterations = 100    # total number of times the algorithm will iterate
k= 3    # number of clusters

centroids = random_centroids(data, k)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    labels = get_labels(data, centroids)    # figure which cluster each data goes to
    centroids = new_centroids(data, labels, k)  #update centroids
    plot_clusters(data, labels, centroids, iteration)
    iteration += 1

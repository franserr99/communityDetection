import pandas as pd
import os

# Read in the data

"""
for i in range(1,5):
    if i == 1:
        iter_max = 6
    else:
        iter_max = 7
        
    for j in range(1,iter_max):
        df = pd.read_csv('Cisco_22_networks/dir_20_graphs/dir_day'+str(i)+'/out'+str(i)+'_'+str(j)+'.txt.gz', compression='gzip', sep="\t", header=None, names=["graph","src", "dst", "weight"], dtype={"graph": str, "src": int, "dst": int, "weight": str})
        for graph in df.graph.unique():
            df_g = df.query("graph == @graph ")
            
            # get unique nodes
            nodes_src = df_g["src"].unique()
            nodes_dst = df_g["dst"].unique()

            nodes = list(set(nodes_src) | set(nodes_dst))
            # map nodes to new index
            node_map = dict(zip(nodes,range(len(nodes))))
            df_g["src"] = df_g["src"].map(node_map)
            df_g["dst"] = df_g["dst"].map(node_map)
            df_g.dropna(inplace=True)
            

            df_g.to_csv('graphs/' + str(graph) + "_" + str(i) + '.csv',mode='a', sep="\t", header=None, index=False)

directory = "graphs/"
for file in os.listdir(directory):
    file = os.path.join(directory, file)
    df = pd.read_csv(file, sep="\t", header=None, names=["graph","src", "dst", "weight"], dtype={"graph": str, "src": float, "dst": float, "weight": str})
    # get unique nodes
    nodes = df["src"].unique()

    # map nodes to new index
    node_map = dict(zip(nodes,range(len(nodes))))
    df["src"] = df["src"].map(node_map)
    df["dst"] = df["dst"].map(node_map)
    df.dropna(inplace=True)
    df.astype({'src': 'int32', 'dst': 'int32'}).dtypes
    print(df)

    df.to_csv(file, sep="\t", header=None, index=False,mode="w",dtype={'src': 'int32', 'dst': 'int32'})

 """

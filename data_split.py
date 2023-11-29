import pandas as pd

# Read in the data
for i in range(1,5):
    if i == 1:
        iter_max = 6
    else:
        iter_max = 7
    
    for j in range(1,iter_max):
        df = pd.read_csv('communityDetection/Cisco_22_networks/dir_20_graphs/dir_day'+str(i)+'/out'+str(i)+'_'+str(j)+'.txt.gz', compression='gzip', sep="\t", header=None, names=["graph","src", "dst", "weight"], dtype={"graph": str, "src": int, "dst": int, "weight": str})
        for graph in df.graph.unique():
            df_g = df.query("graph == @graph ")
            
            # update index of source and destination nodes
            df_g["src"] = df_g["src"].map(lambda x: x-1)
            df_g["dst"] = df_g["dst"].map(lambda x: x-1)
            df_g.to_csv('communityDetection/graphs/' + str(graph) + "_" + str(i) + '.csv',mode='a', sep="\t", header=None, index=False)
        
        
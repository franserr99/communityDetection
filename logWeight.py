import numpy as np
from collections import defaultdict

# Example usage
file_path = "out1_1.txt" # change if need be
with open(file_path, 'r') as file:
    data = file.readlines()

# dict for edges
edge = defaultdict(int)

for line in data:

    parts = line.split()
    graph = parts[0]
    client = parts[1]
    server = parts[2]
    connections = parts[3]

    # splitting up connection and port
    for connection in connections.split(','):
        _, count = connection.split('-')
        key = (graph, client, server)
        edge[key] += int(count)

# Does Log Transformation
packet = np.array(list(edge.values()))
logPacket = np.log(packet + 1)

# normalized data
# the 0.8 is a tuning factor, raise or lower to increase weight (not really important)
weightLog = 0.5 * logPacket

# results
for edge, weight in zip(edge.keys(), weightLog):
    print(f"Edge {edge}: Log Weight {round(weight,4)}")

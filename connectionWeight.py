from collections import defaultdict

# Reads from file
file_path = "out1_1.txt" # change if need be
with open(file_path, 'r') as file:
    data = file.readlines()

# Dict to store weights
edge = defaultdict(set)

# Parse the input data
for line in data:

    parts = line.split()
    graph = parts[0]
    client = parts[1]
    server = parts[2]
    connections = parts[3]

    # Counts connections and uses as weight, probably an easier way of doing this
    for connection in connections.split(','):
        key = (graph, client, server)
        edge[key].add(connection)

# Print weight based on amount of connections
for edge, connections in edge.items():
    totalWeight = len(connections)
    print(f"Edge {edge}: Weight {totalWeight}")

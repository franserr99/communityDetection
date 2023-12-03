from collections import defaultdict

# Reads from file
file_path = "out1_1.txt" # change if need be
with open(file_path, 'r') as file:
    data = file.readlines()

# Dict to store weights
avgWeights = defaultdict(float)

# Dict for connections and dict for packets
edgeConnections = defaultdict(int)
edgePackets = defaultdict(int)

# Splitting data into smaller parts parts
for line in data:
    parts = line.split()

    graph = parts[0]
    client = parts[1]
    server = parts[2]
    connections = parts[3]

    # Separates the final column by commas ex: 1p6-22,1p17-4,2p6-12,3p6-12,4p6-12,5p6-12
    pairs = connections.split(',')

    for pair in pairs:
        _, count = pair.split('-')      # split by hyphen
        key = (graph, client, server)
        edgeConnections[key] += 1            # gets the amount of connections
        edgePackets[key] += int(count)     # gets the sum of packets

# Calculates the average weight from sum of packets and amount of connection
for key, count in edgeConnections.items():
    avgWeights[key] = edgePackets[key] / count

# Calculate Tukey's Biweight Midvariance (uses median rather than std.)
sortedWeights = sorted(avgWeights.values())
medianWeight = sortedWeights[len(sortedWeights) // 2]

# the constants 9 and the median_weights are all adjustable sicne it is meant to be tuning factors.
# pretty much the denominator is completely changable to whatever we want to help normalize the data.
biweight = {}
for edge, weight in avgWeights.items():
    biweight[edge] = (weight - medianWeight) / (9 * medianWeight)


# Outputs the average weight and normalized weight based on biweight midvariance
for edge, weight in biweight.items():
    print(f"Edge {edge}: Average Weight {round(avgWeights[edge], 3)}, Normalized Weight {round(weight,3)}")

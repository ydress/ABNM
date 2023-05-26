import numpy as np
import networkx as nx
from statistics import mean
import random
import matplotlib.pyplot as plt
import math
import sys
from collections import defaultdict, namedtuple
from sklearn.neighbors import BallTree
import seaborn as sns

import heapq
from typing import List, Tuple

def generate_nearest_neighbors_graph(G: nx.Graph, k: int):
    # Create a list of node coordinates and ids
    node_coords = []
    node_ids = []
    for node in G.nodes(data=True):
        node_coords.append(node[1]['coor'])
        node_ids.append(node[0])

    # Build a BallTree
    tree = BallTree(np.array(node_coords))

    # Query the tree for the k nearest neighbors for each node
    dist, ind = tree.query(node_coords, k+1)  # k+1 because a node is its own nearest neighbor

    # Create a dictionary {node_id: [neighbor_ids]}
    neighbors_dict = {node_ids[i]: [node_ids[j] for j in ind[i] if j != i] for i in range(len(node_ids))}

    return neighbors_dict

#coordinates: List[Tuple[float, float]]
def k_closest_points(G: nx.Graph, target: Tuple[float, float], k: int) -> List[Tuple[float, float]]:
    max_heap = []
    coors_list = []
    node_obj = namedtuple("Node", "id, x, y")
    for node in G.nodes(data=True):
            x,y = node[1]['coor']
            id = node[0]
            cur_node = node_obj(id=id, x=x, y=y)
            coors_list.append(cur_node)
    #for coordinate in coordinates:
    for node in coors_list:
        # Calculate Euclidean distance.
        distance = math.sqrt((node.x - target[0])**2 + (node.y - target[1])**2)
        
        # If we have fewer than k items in the heap, we push the new item inside.
        if len(max_heap) < k:
            heapq.heappush(max_heap, (-distance, node.id))
        else:
            # If the new item is closer than the item with the largest distance in the heap,
            # we pop the heap and push the new item inside.
            if distance < -max_heap[0][0]:
                heapq.heappop(max_heap)
                heapq.heappush(max_heap, (-distance, node.id))

    return [coordinate for distance, coordinate in max_heap]

# from INA tutorial
def info(G, fast=False):
    def avg_shortest_path(G):
        random_nodes = random.sample(list(G.nodes()), 40)
        result = []
        for node in random_nodes:
            shortest_path_lengths = nx.shortest_path_length(G, source=node)
            result.append(mean(shortest_path_lengths.values()))
        return mean(result)
    print("{:>12s} | '{:s}'".format('Graph', G.name))

    n = G.number_of_nodes()
    m = G.number_of_edges()

    print("{:>12s} | {:,d} ({:,d})".format('Nodes', n, nx.number_of_isolates(G)))
    print("{:>12s} | {:,d} ({:,d})".format('Edges', m, nx.number_of_selfloops(G)))
    print("{:>12s} | {:.10f} ".format('Density', nx.density(G)))
    print("{:>12s} | {:.2f} ({:,d})".format('Degree', 2 * m / n, max([k for _, k in G.degree()])))
    print("{:>12s} | {:.2f} ".format('Avg_dist', avg_shortest_path(G)))

    C = sorted(nx.connected_components(G), key=len, reverse=True)

    print("{:>12s} | {:.1f}% ({:,d})".format('LCC', 100 * len(C[0]) / n, len(C)))

    if isinstance(G, nx.MultiGraph):
        G = nx.Graph(G)

    print("{:>12s} | {:.4f}".format('Clustering', nx.average_clustering(G)))
    print()
    return G

def closestPairs(coordinates, n):

    # List of pairs to store points on plane
    points = [(coordinates[i][0], coordinates[i][1]) for i in range(n)]

    # Sort them according to their x-coordinates
    points.sort()

    # Minimum distance b/w points seen so far
    d = sys.maxsize

    # Keeping the points in increasing order
    st = set()
    st.add(points[0])
    neighbors = defaultdict(list)

    for i in range(1, n):
        l = set([p for p in st if p[0] >= points[i][0]-d and p[1] >= points[i][1]-d])
        r = set([p for p in st if p[0] <= points[i][0]+d and p[1] <= points[i][1]+d])
        intersection = l & r
        if len(intersection) == 0:
            continue

        for val in intersection:
            dis = math.pow(points[i][0] - val[0], 2) + math.pow(points[i][1] - val[1], 2)

            # Updating the minimum distance dis
            if d > dis:
                d = dis
            # neighbors[val].append(points[i])
            if len(neighbors[val]) < 10:
                neighbors[val].append(points[i])
                neighbors[points[i]].append(val)
            else:
                # Check if the current point is closer than the existing neighbors
                max_distance = max([math.pow(val[0] - p[0], 2) + math.pow(val[1] - p[1], 2) for p in neighbors[val]])
                if dis < max_distance:
                    # Remove the neighbor with maximum distance
                    max_index = [math.pow(val[0] - p[0], 2) + math.pow(val[1] - p[1], 2) for p in neighbors[val]].index(max_distance)

                    max_neighbor = neighbors[val][max_index]

                    del neighbors[val][max_index]
                    del neighbors[max_neighbor][neighbors[max_neighbor].index(val)]
                    # Add the current point as a neighbor
                    neighbors[val].append(points[i])
                    neighbors[points[i]].append(val)
        st.add(points[i])

    return neighbors

def visualize(G):
    pos = nx.spring_layout(G) 
    nx.draw_networkx(G, pos=pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()


def deg_distribution_plot(G):
    # Compute the degree distribution p_k
    degrees = [k for n, k in G.degree()]
    degree_distribution = [[k, degrees.count(k)/G.number_of_nodes()] for k in degrees]
    data = np.array(degree_distribution).T
    # Plot p_k on a log-log plot (Scatterplot)
    # x = k, y = p_k
    sns.scatterplot(x = data[0], y=data[1])
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel("Degree (k)")
    plt.ylabel("Degree Distripution (p_k)")
                                                        
    #plt.loglog(x = data[0], y=data[1], marker=".", linestyle="None")
    #plt.show()
    return plt
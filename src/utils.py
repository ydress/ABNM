import numpy as np
import networkx as nx
from statistics import mean
import random
import matplotlib.pyplot as plt


def similiarity_diff(event_vec, node_vec):
    return np.linalg.norm(event_vec - node_vec)


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


def visualize(G):
    pos = nx.spring_layout(G) 
    nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()


def deg_distribution_plot(G):
    plt.xlabel("Degree")                                                            
    plt.ylabel("Frequency")                                                         
    plt.loglog(nx.degree_histogram(G), marker=".", linestyle="None")
    plt.show()   
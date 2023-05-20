import networkx as nx
import numpy as np
import itertools

from constants import VECTOR_SIZE
from utils import similiarity_diff, info, visualize, deg_distribution_plot


def simulate_event(G, event_vec):
    similarity_scores = []
    for node in G.nodes():
        node_vec = G.nodes[node]['vector']
        similarity = similiarity_diff(event_vec, node_vec)
        similarity_scores.append(similarity)
    # num_selected = int(len(G) * percentage_of_most_sim / 100)
    num_selected = 5
    selected_nodes = np.argsort(similarity_scores)[:num_selected]
    # Create links between the event node and selected nodes
    edges_to_be_added = list(itertools.combinations(selected_nodes, 2))
    G.add_edges_from(edges_to_be_added)


def generate_graph(num_nodes, avg_deg):
    target_m = avg_deg * num_nodes
    G = nx.Graph()

    # Initialize vectors of each node
    node_vectors = np.random.rand(num_nodes, VECTOR_SIZE)
    for i in range(num_nodes):
        G.add_node(i, vector=node_vectors[i])

    while len(G.edges()) < target_m:
        simulate_event(G, np.random.rand(VECTOR_SIZE))

    return G


G = generate_graph(50, 5)
info(G)
visualize(G)
deg_distribution_plot(G)




import networkx as nx
import numpy as np
import itertools
import random

from utils import similiarity_diff, info, visualize, deg_distribution_plot
from node_generation_utils import generate_graph_from_personality_data


def simulate_event(G, G_nodes_list):
    organizer_node = random.sample(G_nodes_list, 1)[0] 
    event_vec = G.nodes[organizer_node]['vector']
    sampled_nodes = random.sample(G_nodes_list, 40)
    similarity_scores = []
    for node in sampled_nodes:
        node_vec = G.nodes[node]['vector']
        similarity = similiarity_diff(event_vec, node_vec)
        similarity_scores.append(similarity)
    # num_selected = int(len(G) * percentage_of_most_sim / 100)
    num_selected = 3
    selected_nodes = np.argsort(similarity_scores)[:num_selected]
    # Create links between the event node and selected nodes
    edges_to_be_added = list(itertools.combinations(selected_nodes, 2))
    G.add_edges_from(edges_to_be_added)


def generate_graph(num_nodes=None, avg_deg=2):
    target_m = avg_deg * num_nodes / 2 
    G = generate_graph_from_personality_data(num_nodes=num_nodes)
    len_vec = len(G.nodes[1]['vector'])
    print (f'creating G with {len(G)} nodes and feature vector of size {len_vec}')

    G_nodes_list = list(G.nodes())

    # # Sample the desired number of nodes
    # sampled_nodes = random.sample(all_nodes, num_samples)

    while len(G.edges()) < target_m:
        simulate_event(G, G_nodes_list)

    return G


G = generate_graph(num_nodes=1000, avg_deg=2)
info(G)
visualize(G)
# deg_distribution_plot(G)




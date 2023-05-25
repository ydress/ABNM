import networkx as nx
import numpy as np
import itertools
import random
import copy
import bisect
import argparse


from utils import info, visualize, deg_distribution_plot, closestPairs, k_closest_points
from node_generation_utils import generate_graph_from_personality_data, generate_graph_from_random_data
from constants import W_PREF, SIGMA


# TODO decide distribution of distances
def generate_distance():
    mean = 0
    sigma = SIGMA
    sample_size = 1
    samples = np.random.normal(mean, sigma, sample_size)
    return samples[0]


# x == 0 , y == 1 -> sorted tuples by axis (node_id, (x,y)), dict(node_id : idx)
def precompute_axis(G, axis):
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['coor'][axis])
    sorted_nodes = [ (node_id, data['coor'][axis]) for node_id, data in sorted_nodes]
    sorted_nodes_idxs = dict()
    for idx, node in enumerate(sorted_nodes):
        node_id = node[0]
        sorted_nodes_idxs[node_id] = idx
    return sorted_nodes, sorted_nodes_idxs


def generate_graph(num_nodes, min_deg):
    # G, similarity_function = generate_graph_from_personality_data(num_nodes=num_nodes)
    G, similarity_function = generate_graph_from_random_data(num_nodes=num_nodes)
    

    node_neighs_dict = dict()
    nearest = min_deg**2
    for node in G.nodes(data=True):
        neighs_list = k_closest_points(G=G, target=node[1]['coor'], k=nearest)
        node_id = node[0]
        node_neighs_dict[node_id] = neighs_list

    sum_degs = 0

    nodes = list(G.nodes())
    shuffled_nodes = copy.copy(nodes)
    random.shuffle(shuffled_nodes)

    sorted_nodes_x, sorted_nodes_idxs_x = precompute_axis(G, 0)
    sorted_nodes_y, sorted_nodes_idxs_y = precompute_axis(G, 1)

    # travel from node on y axis -> destination node
    def travel_axis(node, axis):
        if axis == 0:
            percents = [perc for node_id, perc in sorted_nodes_x]
            idx_in_list = sorted_nodes_idxs_x[node]
            _, perc = sorted_nodes_x[idx_in_list]
        else:
            percents = [perc for node_id, perc in sorted_nodes_y]
            idx_in_list = sorted_nodes_idxs_y[node]
            _, perc = sorted_nodes_y[idx_in_list]
        dist = generate_distance()
        target_perc = perc + dist
        if dist >= 0:
            lower_bound_index = bisect.bisect_left(percents, target_perc)
        if dist < 0:
            lower_bound_index = bisect.bisect_right(percents, target_perc)
        if len(sorted_nodes_x) == lower_bound_index: lower_bound_index -= 1
        if axis == 0: return sorted_nodes_x[lower_bound_index][0]
        else: return sorted_nodes_y[lower_bound_index][0]
        
    def neighs(node):
        return node_neighs_dict[node]

    def sample_nodes_from_destination(traveler_node, destination_node, min_deg):
        neigbours = neighs(destination_node)
        P_attach_list = []
        for node in neigbours:
            if node == traveler_node: 
                P_attach_list.append(0)
            else:
                w_sim = 1 - W_PREF
                sim = similarity_function(G.nodes[traveler_node]['vector'], G.nodes[node]['vector'])
                if not sum_degs:
                    P_deg = 1
                else:
                    P_deg = G.degree[node] / sum_degs
                P_attach = w_sim * sim + W_PREF * P_deg
                P_attach_list.append(P_attach)
        weights = P_attach_list
        sampled_indices = random.choices(range(len(P_attach_list)), weights=weights, k=min_deg)
        sampled_nodes = []
        for idx in sampled_indices: 
            sampled_nodes.append(neigbours[idx])
        return sampled_nodes

    # MAIN
    for node in shuffled_nodes:
        axis = random.choice([0, 1])
        destination_node = travel_axis(node, axis)
        sampled_nodes = sample_nodes_from_destination(node, destination_node, min_deg)
        for sampled_node in sampled_nodes:
            G.add_edge(node, sampled_node)
            sum_degs += 2
    return G


def parse_args():
    parser = argparse.ArgumentParser(description="Graph Generator Script")
    parser.add_argument(
        "-n",
        "--numnodes",
        type=int,
        default=100,
        help="Number of Nodes in Graph",
    )
    parser.add_argument(
        "-d",
        "--mindeg",
        type=int,
        default=2,
        help="minimum degree of nodes",
    )
    parser.add_argument(
        "-g",
        "--gridsearch",
        action="store_true"
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    ARGS = parse_args()
    if ARGS.gridsearch: 
        for param_nodes in [1500]:
            for param_w_pref in [1]:
                for param_sigma in [0.1]:
                    for param_min_deg in [25]:
                        print(
                        f"""
                        N: {param_nodes}, MIN_DEG: {param_min_deg},
                        W_PREF: {param_w_pref},
                        SIGMA: {param_sigma}
                        """)
                        W_PREF = param_w_pref
                        SIGMA = param_sigma
                        G = generate_graph(num_nodes=param_nodes, min_deg=param_min_deg)
                        info(G)
                        visualize(G)
                        deg_distribution_plot(G)
    else:
        #G = generate_graph(num_nodes=100, min_deg=2)
        G = generate_graph(num_nodes=ARGS.numnodes, min_deg=ARGS.mindeg)
        info(G)
        # visualize(G)
        deg_distribution_plot(G)
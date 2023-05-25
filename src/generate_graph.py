import networkx as nx
import numpy as np
import itertools
import random
import copy
import bisect
import argparse


from utils import info, visualize, deg_distribution_plot, closestPairs, k_closest_points
from node_generation_utils import generate_graph_from_personality_data
from constants import w_pref


# TODO decide distribution of distances
def generate_distance():
    mean = 0
    sigma = 0.01
    sample_size = 1
    samples = np.random.normal(mean, sigma, sample_size)
    return samples[0]


def precompute_y(G):
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['coor'][1])
    max_y = sorted_nodes[0][1]['coor'][1]
    min_y = max_y
    for node in sorted_nodes:
        y = node[1]['coor'][1]
        max_y = max ( max_y , y )
        min_y = min ( min_y , y )
    total_y_len = max_y - min_y 
    # map every node to percentual distance from begining to end
    # [ (node_id, perc_dist) ]
    percentual_dist_list = []
    # [ {node_id: index in perc_dist_list} ]
    percentual_dist_dict = dict()
    for node in sorted_nodes:
        node_id = node[0]
        y = node[1]['coor'][1]
        percentual_dist = ( y - min_y ) / total_y_len
        percentual_dist_list.append( (node_id, percentual_dist) )
        percentual_dist_dict[node_id] = len(percentual_dist_list) - 1
    return percentual_dist_list, percentual_dist_dict


def generate_graph(num_nodes, min_deg):
    G, similarity_function = generate_graph_from_personality_data(num_nodes=num_nodes)
    #def precompute_node_neighs_dict():
    #    coors_list = []
    #    for node in G.nodes(data=True):
    #        (x,y) = (node[1]['coor'])
    #        coors_list.append((x,y))
    #    return closestPairs(coors_list, len(G))
    #TODO node id should be key for dict
    #node_neighs_dict = precompute_node_neighs_dict()

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

    percentual_dist_list_y, percentual_dist_dict_y = precompute_y(G)
    percents_y = [y for node_id, y in percentual_dist_list_y]

    # travel from node on y axis -> destination node
    def travel_y(node):
        idx_in_list = percentual_dist_dict_y[node]
        _, perc = percentual_dist_list_y[idx_in_list]
        dist = generate_distance()
        target_perc = perc + dist
        if dist >= 0:
            lower_bound_index = bisect.bisect_left(percents_y, target_perc)
        if dist < 0:
            lower_bound_index = bisect.bisect_right(percents_y, target_perc)
        if len(percentual_dist_list_y) == lower_bound_index: lower_bound_index -= 1
        return percentual_dist_list_y[lower_bound_index][0]
        
    # TODO precompute neighs (what should be the size? maybe square of min_deg?)
    def neighs(node):
        return node_neighs_dict[node]


    def sample_nodes_from_destination(traveler_node, destination_node, min_deg):
        neigbours = neighs(destination_node)
        P_attach_list = []
        for node in neigbours:
            if node == traveler_node: 
                P_attach_list.append(0)
            else:
                w_sim = 1 - w_pref
                sim = similarity_function(G.nodes[traveler_node]['vector'], G.nodes[node]['vector'])
                if not sum_degs:
                    P_deg = 1
                else:
                    P_deg = G.degree[node] / sum_degs
                P_attach = w_sim * sim + w_pref * P_deg
                P_attach_list.append(P_attach)
        weights = P_attach_list
        sampled_indices = random.choices(range(len(P_attach_list)), weights=weights, k=min_deg)
        sampled_nodes = []
        for idx in sampled_indices: 
            sampled_nodes.append(neigbours[idx])
        return sampled_nodes

    # MAIN
    for node in shuffled_nodes:
        # TODO flip a coin and either travel y or x
        destination_node = travel_y(node) 
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
        exit() 
    #G = generate_graph(num_nodes=100, min_deg=2)
    G = generate_graph(num_nodes=ARGS.numnodes, min_deg=ARGS.mindeg)
    info(G)
    visualize(G)
    deg_distribution_plot(G)
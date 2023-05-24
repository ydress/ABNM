import random
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pycristoforo as pyc



def generate_locations_in_europe(n):
    country_name = "Italy"
    country = pyc.get_shape(country_name)
    points = pyc.geoloc_generation(country, n, country_name)
    return [point['geometry']['coordinates'] for point in points]


# https://www.kaggle.com/datasets/arslanali4343/top-personality-dataset
def generate_graph_from_personality_data(num_nodes):
    csv_file = './datasets/personality_data.csv'
    df = pd.read_csv(csv_file)

    G = nx.Graph()

    cols = ['openness','agreeableness','emotional_stability','conscientiousness','extraversion']
    # also normalize these cols
    data = df[cols].values
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    df[cols] = normalized_data

    coors = generate_locations_in_europe(num_nodes + 10)
    for (index, row), coor in zip(df.iterrows(), coors):
        if num_nodes is not None and index >= num_nodes: break
        vec = []
        for col in cols:
            vec.append(row[col])
        node_id = index + 1
        G.add_node(node_id, coor=np.array(coor), vector=np.array(vec))

    def similarity_function(u_vec,v_vec):
        feat_const = 1
        feat_sim = np.linalg.norm(u_vec - v_vec)
        return feat_const * feat_sim

    return G, similarity_function


if __name__ == "__main__":
    G = generate_graph_from_personality_data(100)
    for node in G.nodes(data=True):
        print("Node ID:", node[0])
        print("vector: ", node[1]['vector'])
        print("coor: ", node[1]['coor'])
        print()

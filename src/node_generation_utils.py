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
    coors = [point['geometry']['coordinates'] for point in points]
    def normalize_coordinates(x, y, min_x, max_x, min_y, max_y):
        normalized_x = (x - min_x) / (max_x - min_x)
        normalized_y = (y - min_y) / (max_y - min_y)
        return normalized_x, normalized_y
    max_x = coors[0][0]
    min_x = max_x
    max_y = coors[0][1]
    min_y = max_y
    for x, y in coors:
        max_x = max( max_x, x )
        min_x = min( min_x, x )
        max_y = max( max_y, y )
        min_y = min( min_y, y )
    normalized_coors = []
    for x, y in coors:
        coor = normalize_coordinates(x, y , min_x, max_x, min_y, max_y)
        normalized_coors.append(coor)
    return normalized_coors


# https://www.kaggle.com/datasets/arslanali4343/top-personality-dataset
def generate_graph_from_personality_data(num_nodes):
    if num_nodes > 1835: exit(1)
    csv_file = './datasets/personality_data.csv'
    df = pd.read_csv(csv_file)

    G = nx.Graph()

    # TODO maybe use other feature v gen
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

    # TODO craft sim f
    def similarity_function(A,B):
        # Use cosine similatrity
        cosine = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
        return (cosine + 1) / 2

    return G, similarity_function


if __name__ == "__main__":
    G = generate_graph_from_personality_data(100)
    for node in G.nodes(data=True):
        print("Node ID:", node[0])
        print("vector: ", node[1]['vector'])
        print("coor: ", node[1]['coor'])
        print()

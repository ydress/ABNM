import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np



# https://www.kaggle.com/datasets/arslanali4343/top-personality-dataset
def generate_graph_from_personality_data(num_nodes=None):
    csv_file = './datasets/personality_data.csv'
    df = pd.read_csv(csv_file)

    G = nx.Graph()

    cols = ['openness','agreeableness','emotional_stability','conscientiousness','extraversion']
    # also normalize these cols
    data = df[cols].values
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    df[cols] = normalized_data

    for index, row in df.iterrows():
        if num_nodes is not None and index >= num_nodes: break
        vec = []
        for col in cols:
            vec.append(row[col])
        node_id = index + 1
        G.add_node(node_id, vector=np.array(vec))
    return G


if __name__ == "__main__":
    G = generate_graph_from_personality_data()
    for node in G.nodes(data=True):
        print("Node ID:", node[0])
        print("vector: ", node[1]['vector'])
        print()

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


# this is stub implementation which disregards homophily
def generate_graph_from_random_data(num_nodes):
    G = nx.Graph()
    coors = generate_locations_in_europe(num_nodes + 10)
    for idx, coor in enumerate(coors):
        node_id = idx + 1
        G.add_node(node_id, coor=np.array(coor), vector=np.array([1]))

    def similarity_function(A,B):
        return 1

    return G, similarity_function


# https://www.kaggle.com/datasets/arslanali4343/top-personality-dataset
def generate_graph_from_personality_data(num_nodes):
    #if num_nodes > 1835: exit(1)
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

    def similarity_function(A,B):
        # Use cosine similatrity
        cosine = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
        return (cosine + 1) / 2

    return G, similarity_function

def generate_features(num_samples):
    # Define the distributions
    age_distribution = pd.Series([0.15, 0.24, 0.19, 0.16, 0.13, 0.11], 
                                 index=['18–24', '25–34', '35–44', '45–54', '55–64', '65+'])
    gender_distribution = pd.Series([0.46, 0.54], index=['male', 'female'])
    religion_distribution = pd.Series([0.319, 0.148, 0.002, 0.271, 0.003, 0.001, 0.129, 0.127], 
                                      index=['Christian', 'Hindu', 'Jewish', 'Muslim', 'Sikh', 'Traditional Spirituality', 'Other Religions', 'No religious affiliation'])
    language_distribution = pd.Series([0.64, 0.17, 0.15, 0.11, 0.099, 0.077, 0.066, 0.065, 0.064, 0.062], 
                                      index=['English', 'Spanish', 'Portuguese', 'French', 'German', 'Indonesian', 'Japanese', 'Vietnamese', 'Arabic', 'Hindi'])
    marital_status_distribution = pd.Series([0.315, 0.514, 0.105, 0.066], 
                                            index=['Single', 'Married', 'Divorced', 'Widowed'])
    profession_distribution = pd.Series([0.122, 0.171, 0.139, 0.178, 0.23, 0.07, 0.09], 
                                        index=['Manager', 'Professional', 'Service', 'Sales and office', 'Student', 'Natural resources construction and maintenance', 'Production transportation and material moving'])
    political_orientation_distribution = pd.Series([0.094, 0.347, 0.181, 0.18, 0.105, 0.08, 0.013], 
                                                  index=['Far Left', 'Left', 'Center Left', 'Center', 'Center Right', 'Right', 'Far Right'])
    interests_distribution = pd.Series([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], 
                                       index=['Brands', 'Celebrities', 'Sports Teams', 'Movies', 'TV Show', 'Games', 'News', 'Organizations'])
    
    # Normalize distributions
    distributions = [age_distribution, gender_distribution, religion_distribution, language_distribution,
                     marital_status_distribution, profession_distribution, political_orientation_distribution,
                     interests_distribution]
    for distribution in distributions:
        distribution /= distribution.sum()

    # Generate samples
    age_samples = np.random.choice(age_distribution.index, p=age_distribution.values, size=num_samples)
    gender_samples = np.random.choice(gender_distribution.index, p=gender_distribution.values, size=num_samples)
    religion_samples = np.random.choice(religion_distribution.index, p=religion_distribution.values, size=num_samples)
    language_samples = np.random.choice(language_distribution.index, p=language_distribution.values, size=num_samples)
    marital_status_samples = np.random.choice(marital_status_distribution.index, p=marital_status_distribution.values, size=num_samples)
    profession_samples = np.random.choice(profession_distribution.index, p=profession_distribution.values, size=num_samples)
    political_orientation_samples = np.random.choice(political_orientation_distribution.index, p=political_orientation_distribution.values, size=num_samples)
    interests_samples = np.random.choice(interests_distribution.index, p=interests_distribution.values, size=num_samples)

    # Combine all features into one DataFrame
    data = pd.DataFrame({
        'age': age_samples,
        'gender': gender_samples,
        'religion': religion_samples,
        'language': language_samples,
        'marital_status': marital_status_samples,
        'profession': profession_samples,
        'political_orientation': political_orientation_samples,
        'interests': interests_samples
    })

    # Convert categorical variables into numerical form for easier processing later on
    for col in data.columns:
        data[col] = data[col].astype('category').cat.codes

    return data

def generate_graph_from_paper(num_nodes):
    #if num_nodes > 1835: exit(1)

    G = nx.Graph()

    # Generate feature vectors
    features_df = generate_features(num_nodes)

    # Normalize the feature vectors
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features_df)
    features_df = pd.DataFrame(normalized_features, columns=features_df.columns)

    # Generate coordinates
    coors = generate_locations_in_europe(num_nodes + 10)

    # Add nodes to the graph
    for (index, row), coor in zip(features_df.iterrows(), coors):
        if num_nodes is not None and index >= num_nodes: break
        vec = row.values
        node_id = index + 1
        G.add_node(node_id, coor=np.array(coor), vector=np.array(vec))

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

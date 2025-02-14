import datetime
import pandas as pd
import json
import requests
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as f
import osmnx as ox
import os
import networkx as nx
from torch_geometric.utils import from_networkx
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import learn_based as lb
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)
# API key
API_KEY = api_keys['Weather_API']['API_key']

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# User input
city_name = "Brussels"
date_time = "2025-02-07 14:00"  # date and time
city_grid_file = "Training Data/city_grid_density.ods"
stib_stops_file = "Training Data/stib_stops.ods"
poi_tags_file = "poi_tags.json"


def read_city_grid(file_path):
    """ Reads and cleans city grid density data """
    df = pd.read_excel(file_path, engine="odf")
    df = df[["min_lat",	"max_lat",	"min_lon", "max_lon", "density_rank"]].dropna()
    return df


def read_stib_stops(file_path):
    """ Reads and cleans STIB stops data """
    df = pd.read_excel(file_path, engine="odf")
    df = df[["stop_lat", "stop_lon", "stop_name"]].dropna()
    return df


def read_poi_tags(file_path):
    """ Reads POI tags and separates names and ranks while keeping their connection """
    with open(file_path, "r") as f:
        poi_data = json.load(f)

    poi_names = []
    poi_ranks = []

    for category in poi_data:
        for name, rank in poi_data[category].items():
            poi_names.append(name)
            poi_ranks.append(rank)

    return poi_names, poi_ranks


def get_weather(city, date_time):
    """Fetches weather data for a given city and datetime."""

    base_forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    base_current_url = "https://api.openweathermap.org/data/2.5/weather"

    # Convert user input to datetime object
    user_date = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M")
    today = datetime.datetime.now()

    # Forecast API can predict up to 5 days ahead
    if user_date >= today and (user_date - today).days <= 5:
        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }
        response = requests.get(base_forecast_url, params=params)

        if response.status_code == 200:
            data = response.json()

            # Find closest forecast time
            closest_forecast = min(data["list"],
                                   key=lambda x: abs(datetime.datetime.fromtimestamp(x["dt"]) - user_date))

            temperature = closest_forecast["main"]["temp"]
            weather_conditions = [weather["main"] for weather in closest_forecast["weather"]]
            is_raining = "Rain" in weather_conditions

            return temperature, is_raining
        else:
            return None, None  # Error in fetching forecast

    elif user_date.date() == today.date():
        # Get current weather if the requested time is today
        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }
        response = requests.get(base_current_url, params=params)

        if response.status_code == 200:
            data = response.json()
            temperature = data["main"]["temp"]
            weather_conditions = [weather["main"] for weather in data["weather"]]
            is_raining = "Rain" in weather_conditions

            return temperature, is_raining
        else:
            return None, None  # Error in fetching current weather

    else:
        return None, None  # Past weather requires a paid plan


def aggregate_poi_ranks(pois, tag_rank_mapping, tag_key='amenity'):
    total_rank = 0
    for poi in pois:
        tag_value = poi.get('tags', {}).get(tag_key)
        if tag_value and tag_value in tag_rank_mapping:
            try:
                total_rank += float(tag_rank_mapping[tag_value])
            except ValueError:
                # If conversion fails, ignore this value.
                pass
    return total_rank


def get_pois(min_lat, min_lon, max_lat, max_lon, poi_type='amenity', timeout=10, tag_rank_mapping=None):
    # the filter is applied on the same key as poi_type.
    query = f"""
    [out:json];
    node[{poi_type}]
      ["{poi_type}"!="bench"]
      ["{poi_type}"!="recycle"]
      ["{poi_type}"!="waste_basket"]
      ["{poi_type}"!="bicycle_parking"]
      ({min_lat},{min_lon},{max_lat},{max_lon});
    out body;
    """
    url = "https://overpass-api.de/api/interpreter"

    try:
        response = requests.post(url, data={'data': query}, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        pois = []

        for element in data['elements']:
            if 'tags' in element:
                poi = {
                    'name': element['tags'].get('name', 'Unnamed POI'),
                    'latitude': element['lat'],
                    'longitude': element['lon'],
                    'type': poi_type,
                    'tags': element['tags']
                }
                pois.append(poi)

        # Optionally, aggregate the popularity rank values if mapping provided.
        popularity_total = None
        if tag_rank_mapping is not None:
            popularity_total = aggregate_poi_ranks(pois, tag_rank_mapping, tag_key=poi_type)

        # For demonstration, print first few POIs.
        # for idx, poi in enumerate(pois[:5], 1):
        #     print(f"{idx}. {poi['name']}")
        #     print(f"   Coordinates: ({poi['latitude']}, {poi['longitude']})")
        #     print(f"   {poi_type.capitalize()}: {poi['tags'].get(poi_type, 'N/A')}\n")

        return pois, popularity_total

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return [], 0


def extract_grid_features(grid, pois_count, temperature, is_raining):

    # Extract density rank from grid data.
    # Use .get() to allow flexibility if grid is a dict; if it's a pd.Series, you could also use grid["density_rank"].
    density_rank = grid.get("density_rank", 0)

    # Construct grid_data with geographic boundaries and density rank.
    grid_data = {
        "min_lat": grid.get("min_lat"),
        "max_lat": grid.get("max_lat"),
        "min_lon": grid.get("min_lon"),
        "max_lon": grid.get("max_lon"),
        "density_rank": density_rank,
    }
    if is_raining:
        rain_val = 1
    else:
        rain_val=0

    if temperature>= 10:
        temp_val = 1
    else:
        temp_val = 0

    # Combine all features into a single dictionary.
    grid_features = {
        "grid_data": grid_data,
        "poi_score": pois_count,
        "density_rank": density_rank,
        "temp": temp_val,
        "rain": rain_val
    }

    return grid_features


def download_road_network(place_name):
    print(f"Downloading road network data for {place_name}...")

    graph_ = ox.graph_from_place(city_name, network_type='all')

    # Get the bounding box of the city
    nodes = ox.graph_to_gdfs(graph_, nodes=True, edges=False)
    north, south, east, west = nodes.union_all().bounds
    radius = 0.01
    # Expand the bounding box
    expanded_north = north + radius
    expanded_south = south - radius
    expanded_east = east + radius
    expanded_west = west - radius

    # Download the expanded graph
    expanded_graph_ = ox.graph_from_bbox((expanded_north, expanded_south, expanded_east, expanded_west), network_type='all')

    return expanded_graph_


def construct_road_graph(road_, bus_stops):
    road_graph = nx.Graph()
    for node, data in road_.nodes(data=True):
        road_graph.add_node(node, **data)
    for u, v, data in road_.edges(data=True):
        road_graph.add_edge(u, v, **data)
    for idx, row in bus_stops.iterrows():
        road_graph.add_node(f'bus_stop_{idx}', y=row['stop_lat'], x=row['stop_lon'], stop_name=row['stop_name'])
    return road_graph


def prepare_data(road_graph, stib_stops_data):
    # Initialize lists to store features and labels
    features = []
    labels = []

    # Iterate over each node in the road graph
    for node, data in road_graph.nodes(data=True):
        # Extract features: e.g., degree of the node (number of connected edges)
        degree = data.get('degree', 0)
        # Add more features as needed

        # Check if the node corresponds to a bus stop
        is_bus_stop = 1 if node in stib_stops_data['stop_name'].values else 0

        # Append features and label
        features.append([degree])  # Add more features here
        labels.append(is_bus_stop)

    # Convert to DataFrame for easier handling
    features_df = pd.DataFrame(features, columns=['degree'])  # Add more feature names
    labels_df = pd.DataFrame(labels, columns=['is_bus_stop'])

    # Combine features and labels into a single DataFrame
    data = pd.concat([features_df, labels_df], axis=1)
    return data


class GCN(torch.nn.Module):
    """Graph Convolutional Network for bus stop prediction"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = f.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def prepare_gnn_data(road_graph, grid_features_list, stib_stops_data):
    """Prepares PyG Data object with node features and labels"""
    # Extract road nodes and coordinates
    road_nodes = [n for n in road_graph.nodes if not str(n).startswith('bus_stop_')]
    road_coords = np.array([[road_graph.nodes[n]['y'], road_graph.nodes[n]['x']] for n in road_nodes])

    # Create labels using nearest neighbor search
    bus_stop_coords = stib_stops_data[['stop_lat', 'stop_lon']].values
    nbrs = NearestNeighbors(n_neighbors=1).fit(road_coords)
    _, indices = nbrs.kneighbors(bus_stop_coords)
    labels = np.zeros(len(road_nodes), dtype=int)
    labels[indices.flatten()] = 1

    # Create node features
    features = []
    for node in road_nodes:
        data = road_graph.nodes[node]
        lat, lon = data['y'], data['x']
        degree = road_graph.degree[node]

        # Find grid features
        grid_feature = next((gf for gf in grid_features_list
                             if gf['grid_data']['min_lat'] <= lat <= gf['grid_data']['max_lat'] and
                             gf['grid_data']['min_lon'] <= lon <= gf['grid_data']['max_lon']), None)

        features.append([
            lat, lon, degree,
            grid_feature['density_rank'] if grid_feature else 0,
            grid_feature['poi_score'] if grid_feature else 0,
            1 if degree > 2 else 0
        ])

    # Normalize features
    features = np.array(features, dtype=np.float32)
    features[:, 0] = (features[:, 0] - features[:, 0].min()) / (features[:, 0].max() - features[:, 0].min())
    features[:, 1] = (features[:, 1] - features[:, 1].min()) / (features[:, 1].max() - features[:, 1].min())

    # Create PyG Data object
    edge_index = from_networkx(road_graph.subgraph(road_nodes)).edge_index
    return Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long)
    )


def train_model(data, epochs=200):
    """Trains the GNN model"""
    model = GCN(input_dim=6, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Create train/test masks
    indices = np.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze()[data.train_mask], data.y.float()[data.train_mask])
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(out.squeeze()) > 0.5).float()
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        if epoch % 20 == 0:
            print(f'Epoch {epoch:3d} | Loss: {loss:.4f} | Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}')

    return model


def predict_new_stops(model, data, road_graph, stib_stops, min_distance=500):
    """Generates predictions and filters valid locations"""
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data)).squeeze().numpy()

    # Create candidate DataFrame
    road_nodes = [n for n in road_graph.nodes if not str(n).startswith('bus_stop_')]
    df = pd.DataFrame({
        'node': road_nodes,
        'lat': [road_graph.nodes[n]['y'] for n in road_nodes],
        'lon': [road_graph.nodes[n]['x'] for n in road_nodes],
        'degree': [road_graph.degree[n] for n in road_nodes],
        'probability': probs
    })

    # Filter candidates
    valid_candidates = df[
        (df['degree'] <= 2) &  # Exclude junctions
        (df['probability'] > 0.7)  # High probability
        ].sort_values('probability', ascending=False)

    # Calculate distances to existing stops
    existing_coords = stib_stops[['stop_lat', 'stop_lon']].values
    candidate_coords = valid_candidates[['lat', 'lon']].values
    distances = cdist(candidate_coords, existing_coords) * 111000  # Convert degrees to meters
    valid_candidates['min_distance'] = np.min(distances, axis=1)

    return valid_candidates[valid_candidates['min_distance'] > min_distance]

def main():
    global features
    city_grid_data = read_city_grid(city_grid_file)
    stib_stops_data = read_stib_stops(stib_stops_file)
    poi_names, poi_ranks = read_poi_tags(poi_tags_file)
    tag_rank_mapping = dict(zip(poi_names, poi_ranks))
    temperature, is_raining = get_weather(city_name, date_time)

    road_ = download_road_network(city_name)
    road_graph = construct_road_graph(road_, stib_stops_data)

    data = prepare_data(road_graph, stib_stops_data)

    # Process city grids and collect features
    grid_features_list = []
    for _, grid in city_grid_data.iterrows():
        pois, poi_count = get_pois(
            grid["min_lat"], grid["min_lon"], grid["max_lat"], grid["max_lon"],
            'amenity', tag_rank_mapping=tag_rank_mapping
        )
        grid_features = extract_grid_features(grid, poi_count, temperature, is_raining)
        grid_features_list.append(grid_features)

    # Prepare GNN data
    data = prepare_gnn_data(road_graph, grid_features_list, stib_stops_data)

    # Train model
    model = train_model(data)

    # Generate predictions
    new_stops = predict_new_stops(model, data, road_graph, stib_stops_data)

    # Display results
    print("\nRecommended new bus stop locations:")
    print(new_stops[['lat', 'lon', 'probability', 'min_distance']].head(10))

    print("Extracted Grid Features:")
    print(features)

    print(f"Graph {road_graph}")
    print(f"Road {road_}")

    if temperature is not None:
        print(f"\nWeather in {city_name} on {date_time}:")
        print(f"Temperature: {temperature}°C")
        print(f"Raining: {'Yes' if is_raining else 'No'}")
    else:
        print("\nError: Unable to fetch weather for this date (historical data requires a paid plan).")

    print("City Grid Data Sample:")
    print(city_grid_data.head())

    print("\nSTIB Stops Data Sample:")
    print(stib_stops_data.head())

    print("POI Names:")
    print(poi_names[:5])

    print("\nPOI Ranks:")
    print(poi_ranks[:5])

if __name__ == "__main__":
    main()



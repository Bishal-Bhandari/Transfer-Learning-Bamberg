import datetime
import pandas as pd
import json
import requests
import torch
import numpy as np
import osmnx as ox
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from shapely.geometry import Point, LineString, Polygon
import networkx as nx
import geopandas as gpd
import folium
from openpyxl import Workbook
import pickle

# Load API keys
with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)
API_KEY = api_keys['Weather_API']['API_key']

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
CITY_NAME = "Brussels"
DATE_TIME = "2025-02-15 11:00"
GRID_FILE = "Training Data/city_grid_density.ods"
STOPS_FILE = "Training Data/stib_stops.ods"
POI_TAGS_FILE = "poi_tags.json"
MODEL_SAVE_PATH = "bus_stop_predictor.pkl"  # Updated to .pkl
JUNCTION_BUFFER = 50  # meters
CELL_SIZE = 500  # meters


# Existing functions from the user's code (unchanged)
def read_city_grid(file_path):
    df = pd.read_excel(file_path, engine="odf")
    df = df[["min_lat", "max_lat", "min_lon", "max_lon", "density_rank"]].dropna()
    return df


def read_stib_stops(file_path):
    df = pd.read_excel(file_path, engine="odf")
    df = df[["stop_lat", "stop_lon", "stop_name"]].dropna()
    return df


def read_poi_tags(file_path):
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
    base_forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    base_current_url = "https://api.openweathermap.org/data/2.5/weather"
    user_date = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M")
    today = datetime.datetime.now()
    if user_date >= today and (user_date - today).days <= 5:
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get(base_forecast_url, params=params)
        if response.status_code == 200:
            data = response.json()
            closest_forecast = min(data["list"],
                                   key=lambda x: abs(datetime.datetime.fromtimestamp(x["dt"]) - user_date))
            temperature = closest_forecast["main"]["temp"]
            weather_conditions = [weather["main"] for weather in closest_forecast["weather"]]
            is_raining = "Rain" in weather_conditions
            return temperature, is_raining
        else:
            return None, None
    elif user_date.date() == today.date():
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get(base_current_url, params=params)
        if response.status_code == 200:
            data = response.json()
            temperature = data["main"]["temp"]
            weather_conditions = [weather["main"] for weather in data["weather"]]
            is_raining = "Rain" in weather_conditions
            return temperature, is_raining
        else:
            return None, None
    else:
        return None, None


def aggregate_poi_ranks(pois, tag_rank_mapping, tag_key='amenity'):
    total_rank = 0
    for poi in pois:
        tag_value = poi.get('tags', {}).get(tag_key)
        if tag_value and tag_value in tag_rank_mapping:
            try:
                total_rank += float(tag_rank_mapping[tag_value])
            except ValueError:
                pass
    return total_rank


def get_pois(min_lat, min_lon, max_lat, max_lon, poi_type='amenity', timeout=10, tag_rank_mapping=None):
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
        popularity_total = None
        if tag_rank_mapping is not None:
            popularity_total = aggregate_poi_ranks(pois, tag_rank_mapping, tag_key=poi_type)
        return pois, popularity_total
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return [], 0


def download_road_network(place_name):
    print(f"Downloading road network data for {place_name}...")
    graph_ = ox.graph_from_place(place_name, network_type='drive', simplify=False)
    nodes = ox.graph_to_gdfs(graph_, nodes=True, edges=False)
    north, south, east, west = nodes.union_all().bounds
    radius = 0.1
    expanded_north = north + radius
    expanded_south = south - radius
    expanded_east = east + radius
    expanded_west = west - radius
    expanded_graph_ = ox.graph_from_bbox((expanded_north, expanded_south, expanded_east, expanded_west),
                                         network_type='drive', simplify=False)
    for node_id in expanded_graph_.nodes():
        expanded_graph_.nodes[node_id]['osmid'] = node_id
    u_values = {(u, v, k): u for u, v, k in expanded_graph_.edges(keys=True)}
    v_values = {(u, v, k): v for u, v, k in expanded_graph_.edges(keys=True)}
    nx.set_edge_attributes(expanded_graph_, u_values, 'u')
    nx.set_edge_attributes(expanded_graph_, v_values, 'v')
    return expanded_graph_


def validate_road_data(road_data):
    missing_info = []
    if isinstance(road_data, nx.Graph):
        try:
            nodes, edges = ox.graph_to_gdfs(road_data)
        except Exception as e:
            raise ValueError(f"Error converting graph to GeoDataFrames: {e}")
    else:
        raise ValueError("road_data must be a NetworkX graph.")
    if 'osmid' not in nodes.columns and 'id' not in nodes.columns:
        nodes = nodes.copy()
        nodes['osmid'] = nodes.index
        print("Node identifier not found; using index as 'osmid'.")
    required_node_cols = ['geometry', 'x', 'y']
    for col in required_node_cols:
        if col not in nodes.columns:
            missing_info.append(f"Missing '{col}' in nodes.")
    if 'u' not in edges.columns or 'v' not in edges.columns:
        def extract_uv(row):
            if isinstance(row.name, tuple) and len(row.name) >= 2:
                return pd.Series({'u': row.name[0], 'v': row.name[1]})
            else:
                return pd.Series({'u': None, 'v': None})

        uv = edges.apply(extract_uv, axis=1)
        if uv['u'].isna().all() or uv['v'].isna().all():
            missing_info.append("Missing 'u' and 'v' in edges and could not be inferred.")
        else:
            edges = edges.join(uv)
            print("Edge identifiers 'u' and 'v' were missing; inferred from index.")
    if missing_info:
        raise ValueError("Road data is missing required fields:\n" + "\n".join(missing_info))
    print("Road data validation passed.")
    return nodes, edges


# New functions for GNN prediction

def normalize_features(density, poi_count, temperature, rain):
    """Normalize features based on specified ranges."""
    density_norm = (density - 1) / 9  # 1-10 to [0,1]
    poi_norm = (poi_count - 1) / 999  # 1-1000 to [0,1]
    if temperature < 3:
        temp_norm = 0  # Bad
    elif 3 <= temperature <= 10:
        temp_norm = (temperature - 3) / 7  # Average, map 3-10 to [0,1]
    else:
        temp_norm = 1  # Good
    rain_norm = rain  # Already 0 or 1 (0 = no rain, 1 = raining)
    return np.array([density_norm, poi_norm, temp_norm, rain_norm])


def construct_graph(road_nodes, road_edges, grid, features):
    """Construct a PyTorch Geometric graph for a specific grid."""
    # Filter edges within the grid
    grid_polygon = Polygon([
        (grid['min_lon'], grid['min_lat']),
        (grid['min_lon'], grid['max_lat']),
        (grid['max_lon'], grid['max_lat']),
        (grid['max_lon'], grid['min_lat'])
    ])
    edges_gdf = gpd.GeoDataFrame(road_edges, geometry='geometry')
    edges_in_grid = edges_gdf[edges_gdf.intersects(grid_polygon)].copy()

    # Create a subgraph
    G = nx.Graph()
    for _, edge in edges_in_grid.iterrows():
        u, v = edge['u'], edge['v']
        G.add_edge(u, v, length=edge['length'], speed_limit=edge.get('maxspeed', 50))

    # Add nodes and features
    node_list = list(G.nodes())
    if not node_list:
        return None  # No nodes in this grid

    # Map nodes to indices
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Edge index for PyTorch Geometric
    edge_index = torch.tensor(
        [[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    # Node features: [length, speed_limit, density, poi, temp, rain]
    x = []
    for node in node_list:
        connected_edges = list(G.edges(node, data=True))
        if connected_edges:
            lengths = [edge[2]['length'] for edge in connected_edges]
            speeds = [float(edge[2]['speed_limit']) if edge[2]['speed_limit'].isdigit() else 50.0
                      for edge in connected_edges]
            avg_length = np.mean(lengths)
            avg_speed = np.mean(speeds)
        else:
            avg_length = 0
            avg_speed = 50.0
        node_features = [avg_length, avg_speed] + features.tolist()
        x.append(node_features)

    x = torch.tensor(x, dtype=torch.float)

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    # Identify junctions (nodes with degree > 2)
    degrees = dict(G.degree())
    data.junction_mask = torch.tensor([degrees[node] > 2 for node in node_list], dtype=torch.bool)

    # Store node IDs for mapping back
    data.node_ids = node_list

    return data


class BusStopGNN(nn.Module):
    """Graph Neural Network for bus stop prediction."""

    def __init__(self, input_dim=6, hidden_dim=64):
        super(BusStopGNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc(x)
        return torch.sigmoid(x)


def load_pretrained_model(path):
    """Load the pre-trained GNN model from a .pkl file."""
    with open(path, 'rb') as f:
        model_state = pickle.load(f)
    model = BusStopGNN(input_dim=6)  # Adjust input_dim if your model expects different features
    model.load_state_dict(model_state['model_state_dict'])
    model.eval()
    return model


def predict_bus_stops(model, data, road_nodes, threshold=0.7, min_distance=100):
    """Predict new bus stops, avoiding junctions and ensuring minimum distance."""
    with torch.no_grad():
        probs = model(data).squeeze()

    # Filter out junctions
    valid_mask = ~data.junction_mask & (probs > threshold)
    candidate_indices = torch.where(valid_mask)[0]
    candidate_nodes = [data.node_ids[idx.item()] for idx in candidate_indices]
    candidate_probs = probs[candidate_indices].numpy()

    # Ensure bus stops are on drivable roads and maintain minimum distance
    selected_stops = []
    occupied_coords = set()

    # Sort candidates by probability descending
    sorted_candidates = sorted(zip(candidate_nodes, candidate_probs), key=lambda x: x[1], reverse=True)

    for node_id, prob in sorted_candidates:
        lat, lon = road_nodes.loc[node_id, ['y', 'x']]
        point = Point(lon, lat)
        too_close = False
        for occ_lat, occ_lon in occupied_coords:
            dist = ox.distance.great_circle(occ_lat, occ_lon, lat, lon)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            selected_stops.append((node_id, lat, lon, prob))
            occupied_coords.add((lat, lon))

    return selected_stops


def main():
    # Load data
    city_grid_data = read_city_grid(GRID_FILE)
    stib_stops_data = read_stib_stops(STOPS_FILE)
    poi_names, poi_ranks = read_poi_tags(POI_TAGS_FILE)
    tag_rank_mapping = dict(zip(poi_names, poi_ranks))
    temperature, is_raining = get_weather(CITY_NAME, DATE_TIME)

    if temperature is None:
        print("Error fetching weather data.")
        return

    road_graph = download_road_network(CITY_NAME)
    road_nodes, road_edges = validate_road_data(road_graph)

    # Load pre-trained GNN model
    model = load_pretrained_model(MODEL_SAVE_PATH)

    # Process each grid and predict bus stops
    all_predictions = []

    for _, grid in city_grid_data.iterrows():
        # Extract features
        pois, poi_count = get_pois(
            grid["min_lat"], grid["min_lon"],
            grid["max_lat"], grid["max_lon"],
            'amenity', tag_rank_mapping=tag_rank_mapping
        )
        density = grid['density_rank']
        rain = 1 if is_raining else 0
        features = normalize_features(density, poi_count, temperature, rain)

        # Construct graph for this grid
        graph_data = construct_graph(road_nodes, road_edges, grid, features)
        if graph_data is None:
            print(
                f"No road segments found in grid: {grid['min_lat']}-{grid['max_lat']}, {grid['min_lon']}-{grid['max_lon']}")
            continue

        # Predict bus stops
        bus_stops = predict_bus_stops(model, graph_data, road_nodes)

        # Add grid information to predictions
        for stop in bus_stops:
            node_id, lat, lon, prob = stop
            all_predictions.append({
                'grid_min_lat': grid['min_lat'],
                'grid_max_lat': grid['max_lat'],
                'grid_min_lon': grid['min_lon'],
                'grid_max_lon': grid['max_lon'],
                'node_id': node_id,
                'latitude': lat,
                'longitude': lon,
                'probability': prob
            })

    # Save predictions to ODS
    wb = Workbook()
    ws = wb.active
    ws.append(['Grid Min Lat', 'Grid Max Lat', 'Grid Min Lon', 'Grid Max Lon', 'Node ID', 'Latitude', 'Longitude',
               'Probability'])
    for pred in all_predictions:
        ws.append([
            pred['grid_min_lat'], pred['grid_max_lat'],
            pred['grid_min_lon'], pred['grid_max_lon'],
            pred['node_id'], pred['latitude'], pred['longitude'], pred['probability']
        ])
    wb.save("bus_stop_predictions.ods")
    print("Predictions saved to 'bus_stop_predictions.ods'")

    # Display on Folium map
    m = folium.Map(location=[50.8503, 4.3517], zoom_start=12)  # Center of Brussels
    for pred in all_predictions:
        folium.Marker(
            [pred['latitude'], pred['longitude']],
            popup=f"Prob: {pred['probability']:.2f}"
        ).add_to(m)
    m.save("bus_stop_predictions.html")
    print("Map saved to 'bus_stop_predictions.html'")


if __name__ == "__main__":
    main()
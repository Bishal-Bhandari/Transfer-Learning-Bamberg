import datetime
import geopandas as gpd
import joblib
import pandas as pd
import json
import requests
import torch
import numpy as np
from shapely.geometry.geo import box
from sklearn.metrics.pairwise import haversine_distances
from tenacity import retry, stop_after_attempt, wait_exponential
import osmnx as ox
import folium
from collections import defaultdict
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.nn import SAGEConv, BatchNorm
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys
with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)
API_KEY = api_keys['Weather_API']['API_key']

# Constants
CITY_NAME = "Brussels"
DATE_TIME = "2023-11-05 11:00"  # Updated to a valid date within 5 days
GRID_FILE = "Training Data/city_grid_density.ods"
STOPS_FILE = "Training Data/stib_stops.ods"
POI_TAGS_FILE = "poi_tags.json"
MODEL_SAVE_PATH = "Output/best_bus_stop_model.pth"
SCALER_SAVE_PATH = "Output/bus_stop_scaler.pkl"
DEFAULT_TEMP = 15.0  # Default temperature if API fails
DEFAULT_RAIN = False

class Config:
    CITY_NAME = "Brussels"
    MIN_STOP_DISTANCE = 500  # meters
    PREDICTION_THRESHOLD = 0.5
    ROAD_TYPES = ['motorway', 'trunk', 'primary', 'secondary']


ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.cache_folder = "osmnx_cache"  # Optional custom cache location
tqdm.pandas()


class BusStopPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.predictor = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.predictor(x)


def read_city_grid(file_path):
    required_columns = ["min_lat", "max_lat", "min_lon", "max_lon", "density_rank"]
    df = pd.read_excel(file_path, engine="odf")
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in grid file. Needed: {required_columns}")
    return df[required_columns].dropna()


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
            return None, None
    elif user_date.date() == today.date():
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_pois(min_lat, min_lon, max_lat, max_lon, poi_type='amenity', timeout=100, tag_rank_mapping=None):
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
        popularity_total = aggregate_poi_ranks(pois, tag_rank_mapping, tag_key=poi_type) if tag_rank_mapping else None
        return pois, popularity_total
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return [], 0


def extract_grid_features(grid, pois_count, temperature, is_raining):
    density_rank = grid.get("density_rank", 0)
    grid_data = {
        "min_lat": grid.get("min_lat"),
        "max_lat": grid.get("max_lat"),
        "min_lon": grid.get("min_lon"),
        "max_lon": grid.get("max_lon"),
        "density_rank": density_rank,
    }
    rain_val = 1 if is_raining else 0
    grid_features = {
        "grid_data": grid_data,
        "poi_score": pois_count,
        "density_rank": density_rank,
        "temp": temperature,
        "rain": rain_val
    }
    return grid_features


def download_road_network(place_name):
    logger.info(f"Downloading road network for {place_name}...")

    graph_ = ox.graph_from_place(place_name, network_type='drive', simplify=False)

    # Get the bounding box of the city
    nodes = ox.graph_to_gdfs(graph_, nodes=True, edges=False)
    north, south, east, west = nodes.union_all().bounds
    radius = 0.1
    # Expand the bounding box
    expanded_north = north + radius
    expanded_south = south - radius
    expanded_east = east + radius
    expanded_west = west - radius

    # Download the expanded graph
    expanded_graph_ = ox.graph_from_bbox((expanded_north, expanded_south, expanded_east, expanded_west), network_type='drive', simplify=False)

    # Explicitly add 'osmid' attribute to nodes (matches graph node IDs)
    for node_id in expanded_graph_.nodes():
        expanded_graph_.nodes[node_id]['osmid'] = node_id

    # Explicitly add 'u' and 'v' attributes to edges
    # Add 'u' and 'v' to edges using bulk operations
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
    elif isinstance(road_data, dict):
        if not all(k in road_data for k in ['nodes', 'edges']):
            raise ValueError("Input dict must contain keys 'nodes' and 'edges'.")
        nodes = road_data['nodes']
        edges = road_data['edges']
    else:
        raise ValueError("road_data must be a NetworkX graph or a dict with 'nodes' and 'edges'.")

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


def load_model_and_scaler(model_path, scaler_path):
    state_dict = torch.load(model_path)
    if "predictor.weight" in state_dict:
        state_dict["lin.weight"] = state_dict.pop("predictor.weight")
        state_dict["lin.bias"] = state_dict.pop("predictor.bias")
    if "conv1.lin.weight" in state_dict:
        state_dict["conv1.lin_l.weight"] = state_dict.pop("conv1.lin.weight")
        state_dict["conv1.lin_l.bias"] = state_dict.pop("conv1.bias")
        state_dict["conv1.lin_r.weight"] = state_dict["conv1.lin_l.weight"]
    if "conv2.lin.weight" in state_dict:
        state_dict["conv2.lin_l.weight"] = state_dict.pop("conv2.lin.weight")
        state_dict["conv2.lin_l.bias"] = state_dict.pop("conv2.bias")
        state_dict["conv2.lin_r.weight"] = state_dict["conv2.lin_l.weight"]

    model = BusStopPredictor(in_channels=6, hidden_channels=64, out_channels=1)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler


def process_edges_and_predict(road_graph, city_grid_data, model, scaler, temperature, is_raining):
    edges_gdf = ox.graph_to_gdfs(road_graph, nodes=False, edges=True)
    edge_to_idx = {}
    midpoints = []
    features = []

    # Create geometry for grid cells and set CRS explicitly to EPSG:4326
    city_grid_data['geometry'] = city_grid_data.apply(
        lambda row: box(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']),
        axis=1
    )
    grid_gdf = gpd.GeoDataFrame(city_grid_data, geometry='geometry', crs="EPSG:4326")

    # Compute midpoints for each edge
    edges_gdf['midpoint'] = edges_gdf['geometry'].apply(lambda x: x.interpolate(0.5, normalized=True))
    midpoints_gdf = gpd.GeoDataFrame(geometry=edges_gdf['midpoint'], crs="EPSG:4326")

    # Spatial join (CRS now matches, so no warning)
    joined = gpd.sjoin(midpoints_gdf, grid_gdf, how='left', predicate='within')

    # Process each edge to compute features
    for idx, (_, edge) in enumerate(edges_gdf.iterrows()):
        line = edge['geometry']
        midpoint = line.interpolate(0.5, normalized=True)
        midpoint_coords = (midpoint.y, midpoint.x)

        # Determine the grid cell by checking if the midpoint falls inside
        grid_cell = None
        for _, grid in city_grid_data.iterrows():
            if (grid['min_lat'] <= midpoint_coords[0] <= grid['max_lat'] and
                    grid['min_lon'] <= midpoint_coords[1] <= grid['max_lon']):
                grid_cell = grid
                break
        if grid_cell is None:
            continue

        # Try to obtain 'u', 'v', and 'key' for this edge
        try:
            u = edge['u']
            v = edge['v']
            key = edge['key']
        except KeyError:
            if isinstance(edge.name, tuple) and len(edge.name) >= 3:
                u, v, key = edge.name[0], edge.name[1], edge.name[2]
            else:
                continue

        # Save the index for this processed edge
        edge_to_idx[(u, v, key)] = len(midpoints)

        density_rank = grid_cell['density_rank']
        poi_score = grid_cell['poi_score']
        length = line.length
        degree_u = road_graph.degree(u)
        degree_v = road_graph.degree(v)
        is_junction = 1 if (degree_u > 2 or degree_v > 2) else 0
        rain = 1 if is_raining else 0
        temp = temperature

        feature_vector = [density_rank, poi_score, temp, rain, length, is_junction]
        feature_normalized = scaler.transform([feature_vector])[0]
        midpoints.append(midpoint_coords)
        features.append(feature_normalized)

    # Build adjacency solely for processed edges.
    # Create a mapping from node to list of processed edge indices.
    node_to_edges = defaultdict(list)
    for (u, v, key), idx in edge_to_idx.items():
        node_to_edges[u].append(idx)
        node_to_edges[v].append(idx)

    # For each node, all processed edges incident to that node are connected.
    adjacency = defaultdict(set)
    for node, indices in node_to_edges.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                adjacency[indices[i]].add(indices[j])
                adjacency[indices[j]].add(indices[i])

    # Build the final edge_index tensor
    edge_index_list = [[src, dest] for src in adjacency for dest in adjacency[src]]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        output = model(data.x, data.edge_index)
    probabilities = torch.sigmoid(output.squeeze()).numpy()

    return midpoints, features, probabilities


def filter_and_save_predictions(midpoints, features, probabilities, output_file="Model Data/predicted_stops.ods"):
    selected = []
    candidates = [
        (i, prob, lat, lon)
        for i, (prob, (lat, lon)) in enumerate(zip(probabilities, midpoints))
        if prob > 0.5 and features[i][5] == 0
    ]

    selected = []
    coordinates = []
    for i, prob, lat, lon in sorted(candidates, key=lambda x: -x[1]):
        coord_rad = np.radians([[lat, lon]])
        if any(haversine_distances(coord_rad, np.radians([c]))[0][0] * 6371000 < 500 for c in coordinates):
            continue
        selected.append(i)
        coordinates.append([lat, lon])

    for i, prob in enumerate(probabilities):
        is_junction = features[i][5]
        if prob > 0.5 and is_junction == 0:
            selected.append(i)

    selected_midpoints = [(midpoints[i][0], midpoints[i][1]) for i in selected]
    df = pd.DataFrame(selected_midpoints, columns=['latitude', 'longitude'])
    df.to_excel(output_file, engine="odf")

    m = folium.Map(location=[midpoints[0][0], midpoints[0][1]], zoom_start=12)
    for lat, lon in selected_midpoints:
        folium.Marker([lat, lon], icon=folium.Icon(color='green')).add_to(m)
    m.save("Template/bus_stops_map.html")
    return df


def validate_predictions(predicted_gdf: gpd.GeoDataFrame, existing_stops_gdf: gpd.GeoDataFrame):
    """Compare predictions with existing stops"""
    buffer_distance = 50  # meters
    existing_buffers = existing_stops_gdf.buffer(buffer_distance / 111000)
    matched = predicted_gdf.geometry.apply(
        lambda x: any(x.distance(existing) < buffer_distance for existing in existing_buffers)
    )

    logger.info(f"Prediction validation:")
    logger.info(f"Total predictions: {len(predicted_gdf)}")
    logger.info(f"Matching existing stops: {matched.sum()} ({matched.mean():.1%})")
    return matched


def main():
    city_grid_data = read_city_grid(GRID_FILE)
    stib_stops_data = read_stib_stops(STOPS_FILE)
    poi_names, poi_ranks = read_poi_tags(POI_TAGS_FILE)
    tag_rank_mapping = dict(zip(poi_names, poi_ranks))
    temperature, is_raining = get_weather(CITY_NAME, DATE_TIME)

    poi_scores = []
    for _, grid in city_grid_data.iterrows():

        pois, poi_count = get_pois(
            grid["min_lat"], grid["min_lon"],
            grid["max_lat"], grid["max_lon"],
            'amenity', tag_rank_mapping=tag_rank_mapping
        )

        poi_scores.append(poi_count)
    city_grid_data['poi_score'] = poi_scores

    road_graph = download_road_network(CITY_NAME)
    road_nodes, road_edges = validate_road_data(road_graph)

    model, scaler = load_model_and_scaler(MODEL_SAVE_PATH, "Output/bus_stop_scaler.pkl")

    midpoints, features, probabilities = process_edges_and_predict(
        road_graph, city_grid_data, model, scaler, temperature, is_raining
    )

    df_predictions = filter_and_save_predictions(midpoints, features, probabilities)
    print("Predictions saved to predicted_stops.ods and map.html")

    if temperature is not None:
        print(f"\nWeather in {CITY_NAME} on {DATE_TIME}:")
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

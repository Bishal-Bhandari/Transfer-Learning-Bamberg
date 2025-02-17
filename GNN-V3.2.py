import datetime

import joblib
import pandas as pd
import json
import requests
import torch
import numpy as np
import osmnx as ox
from folium import folium
from pandas._libs.internals import defaultdict
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.nn import SAGEConv, BatchNorm
import torch.nn.functional as F

# Load API keys
with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)
API_KEY = api_keys['Weather_API']['API_key']

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
CITY_NAME = "Brussels"
DATE_TIME = "2025-02-17 11:00"
GRID_FILE = "Training Data/city_grid_density.ods"
STOPS_FILE = "Training Data/stib_stops.ods"
POI_TAGS_FILE = "poi_tags.json"
MODEL_SAVE_PATH = "Output/"
JUNCTION_BUFFER = 50  # meters
CELL_SIZE = 500  # meters


class BusStopPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        return self.lin(x)


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

    # Combine all features into a single dictionary.
    grid_features = {
        "grid_data": grid_data,
        "poi_score": pois_count,
        "density_rank": density_rank,
        "temp": temperature,
        "rain": rain_val
    }

    return grid_features


def download_road_network(place_name):
    print(f"Downloading road network data for {place_name}...")

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
    # If input is a NetworkX graph:
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

    # Check for node identifier
    if 'osmid' not in nodes.columns and 'id' not in nodes.columns:
        # Fallback: use the index as identifier and add a column
        nodes = nodes.copy()
        nodes['osmid'] = nodes.index
        print("Node identifier not found; using index as 'osmid'.")

    # Ensure required node columns exist
    required_node_cols = ['geometry', 'x', 'y']
    for col in required_node_cols:
        if col not in nodes.columns:
            missing_info.append(f"Missing '{col}' in nodes.")

    # For edges, check for 'u' and 'v'
    if 'u' not in edges.columns or 'v' not in edges.columns:
        # Fallback: if edges index is a MultiIndex or tuple, try to extract u and v
        def extract_uv(row):
            # If the index of the row is a tuple of length >=2, use its first two elements.
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

    # If any missing info remains, raise error with details.
    if missing_info:
        raise ValueError("Road data is missing required fields:\n" + "\n".join(missing_info))

    print("Road data validation passed.")
    return nodes, edges


def load_model_and_scaler(model_path, scaler_path):
    model = BusStopPredictor(in_channels=6, hidden_channels=64, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler


def process_edges_and_predict(road_graph, city_grid_data, model, scaler, temperature, is_raining):
    edges_gdf = ox.graph_to_gdfs(road_graph, nodes=False, edges=True)
    edge_to_idx = {}
    midpoints = []
    features = []

    for idx, (_, edge) in enumerate(edges_gdf.iterrows()):
        line = edge['geometry']
        midpoint = line.interpolate(0.5, normalized=True)
        midpoint_coords = (midpoint.y, midpoint.x)

        grid_cell = None
        for _, grid in city_grid_data.iterrows():
            if (grid['min_lat'] <= midpoint_coords[0] <= grid['max_lat'] and
                    grid['min_lon'] <= midpoint_coords[1] <= grid['max_lon']):
                grid_cell = grid
                break
        if grid_cell is None:
            continue

        u, v, key = edge['u'], edge['v'], edge['key']
        edge_to_idx[(u, v, key)] = idx

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

    # Build adjacency
    adjacency = defaultdict(list)
    for node in road_graph.nodes():
        connected_edges = list(road_graph.edges(node, keys=True))
        edge_indices = [edge_to_idx[edge] for edge in connected_edges if edge in edge_to_idx]
        for i in range(len(edge_indices)):
            for j in range(i + 1, len(edge_indices)):
                adjacency[edge_indices[i]].append(edge_indices[j])
                adjacency[edge_indices[j]].append(edge_indices[i])

    edge_index = torch.tensor([[src, dest] for src in adjacency for dest in adjacency[src]],
                              dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        output = model(data.x, data.edge_index)
    probabilities = torch.sigmoid(output.squeeze()).numpy()

    return midpoints, features, probabilities


def filter_and_save_predictions(midpoints, features, probabilities, output_file="predicted_stops.ods"):
    selected = []
    for i, prob in enumerate(probabilities):
        is_junction = features[i][5]
        if prob > 0.5 and is_junction == 0:
            selected.append(i)

    selected_midpoints = [(midpoints[i][0], midpoints[i][1]) for i in selected]
    df = pd.DataFrame(selected_midpoints, columns=['latitude', 'longitude'])
    df.to_excel(output_file, engine="odf")

    # Create map
    m = folium.Map(location=[midpoints[0][0], midpoints[0][1]], zoom_start=12)
    for lat, lon in selected_midpoints:
        folium.Marker([lat, lon], icon=folium.Icon(color='green')).add_to(m)
    m.save("bus_stops_map.html")
    return df


def main():
    global features
    city_grid_data = read_city_grid(GRID_FILE)
    stib_stops_data = read_stib_stops(STOPS_FILE)
    poi_names, poi_ranks = read_poi_tags(POI_TAGS_FILE)
    tag_rank_mapping = dict(zip(poi_names, poi_ranks))
    temperature, is_raining = get_weather(CITY_NAME, DATE_TIME)

    # Process city grids and collect features
    poi_scores = []
    for _, grid in city_grid_data.iterrows():
        pois, poi_count = get_pois(
            grid["min_lat"], grid["min_lon"],
            grid["max_lat"], grid["max_lon"],
            'amenity', tag_rank_mapping=tag_rank_mapping
        )
        poi_scores.append(poi_count)
    city_grid_data['poi_score'] = poi_scores

    # Download and validate road network
    road_graph = download_road_network(CITY_NAME)
    road_nodes, road_edges = validate_road_data(road_graph)

    # Load model and scaler
    model, scaler = load_model_and_scaler(MODEL_SAVE_PATH, "bus_stop_scaler.pkl")

    # Process edges and predict
    midpoints, features, probabilities = process_edges_and_predict(
            road_graph, city_grid_data, model, scaler, temperature, is_raining
    )

    # Filter and save
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
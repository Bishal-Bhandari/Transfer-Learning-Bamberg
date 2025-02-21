import datetime
import matplotlib
import pandas as pd
import json
import requests
import torch
import numpy as np
import osmnx as ox
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from shapely.geometry import Point, LineString
import networkx as nx
from sympy.physics.units import temperature
from torch_geometric.nn import SAGEConv, BatchNorm
import torch.nn as nn
import torch.nn.functional as F
import folium
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm
import logging
import tkinter as tk

matplotlib.use('TkAgg')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load API keys
with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)
API_KEY = api_keys['Weather_API']['API_key']

# Constants
CITY_NAME = "Bamberg"
DATE_TIME = "2025-02-22 11:00"  # Valid date within 5 days
GRID_FILE = "Training Data/city_grid_density_bamberg.ods"
STOPS_FILE = "Training Data/stib_stops.ods"
POI_TAGS_FILE = "poi_tags.json"
MODEL_SAVE_PATH = "Output/best_bus_stop_model.pth"
OUTPUT_FILE = "Model Data/bus_stop_predictions.csv"
DEFAULT_TEMP = 15.0
DEFAULT_RAIN = False


class Config:
    CITY_NAME = "Brussels"
    MIN_STOP_DISTANCE = 500
    PREDICTION_THRESHOLD = 0.7
    ROAD_TYPES = ['motorway', 'trunk', 'primary', 'secondary']
    ALLOWED_HIGHWAYS = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified']


ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.cache_folder = "osmnx_cache"
tqdm.pandas()


class BusStopPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(4, 64)  # Matches checkpoint input dimensions
        self.bn1 = BatchNorm(64)
        self.conv2 = SAGEConv(64, 64)  # Matches hidden layer dimensions
        self.bn2 = BatchNorm(64)
        self.predictor = nn.Linear(64, 1)  # Final layer name matches checkpoint

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.predictor(x)
        return torch.sigmoid(x)


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance in meters between two geographic points."""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 1000  # Convert to meters


def filter_predictions(predictions, min_distance):
    """Filter predictions to ensure minimum distance between stops."""
    filtered = []
    # Sort by descending score to keep highest first
    sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)

    for pred in sorted_preds:
        lat1 = pred['lat']
        lon1 = pred['lon']
        keep = True

        # Check against already kept predictions
        for kept in filtered:
            lat2 = kept['lat']
            lon2 = kept['lon']
            if haversine(lat1, lon1, lat2, lon2) < min_distance:
                keep = False
                break

        if keep:
            filtered.append(pred)

    return filtered


def read_city_grid(file_path):
    """ Reads and cleans city grid density data """
    df = pd.read_excel(file_path, engine="odf")
    df = df[["min_lat", "max_lat", "min_lon", "max_lon", "density_rank"]].dropna()
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
            temp = DEFAULT_TEMP
            rain = DEFAULT_RAIN
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
                # If conversion fails, ignore this value.
                pass
    return total_rank


def get_pois(min_lat, min_lon, max_lat, max_lon, poi_type='amenity', timeout=50, tag_rank_mapping=None):
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
        rain_val = 0

    # Combine all features into a single dictionary.
    grid_features = {
        "grid_data": grid_data,
        "poi_score": pois_count,
        "density_rank": density_rank,
        "temp": temperature,
        "rain": rain_val
    }

    return grid_features


def _road_type_to_numeric(road_type):
    """
    Convert OSM highway types to numeric codes.
    Handles both single values and lists (common in OSM data).
    """
    if isinstance(road_type, list):
        road_type = road_type[0]  # Take first value if multiple exist

    hierarchy = {
        'motorway': 0,
        'motorway_link': 0,
        'trunk': 1,
        'trunk_link': 1,
        'primary': 2,
        'primary_link': 2,
        'secondary': 3,
        'secondary_link': 3,
        'tertiary': 4,
        'tertiary_link': 4,
        'unclassified': 5,
        'residential': 6,
        'service': 7,
        'living_street': 8,
        'pedestrian': 9
    }
    return hierarchy.get(road_type, 5)


def download_road_network(place_name):
    print(f"Downloading road network data for {place_name}...")

    graph_ = ox.graph_from_place(place_name, network_type='drive', simplify=False)

    # Get the bounding box of the city
    nodes = ox.graph_to_gdfs(graph_, nodes=True, edges=False)
    north, south, east, west = nodes.union_all().bounds
    radius = 0
    # Expand the bounding box
    expanded_north = north + radius
    expanded_south = south + radius
    expanded_east = east + radius
    expanded_west = west + radius

    # Download the expanded graph
    expanded_graph_ = ox.graph_from_bbox(
        (expanded_north, expanded_south, expanded_east, expanded_west),
        network_type='drive',
        simplify=False
    )

    # Update each node: preserve 'highway' along with x, y, and type_encoded.
    for node_id in list(expanded_graph_.nodes()):
        attrs = expanded_graph_.nodes[node_id]

        # Ensure coordinates exist. Use fallback to 'lon' and 'lat' if needed.
        x_val = attrs.get('x')
        y_val = attrs.get('y')
        if x_val is None or y_val is None:
            x_val = attrs.get('lon', 0.0)
            y_val = attrs.get('lat', 0.0)

        # Preserve highway attribute (or set a default) before clearing attributes.
        highway_val = attrs.get('highway', 'unclassified')
        type_encoded = _road_type_to_numeric(highway_val)

        # Clear attributes and update with required ones, including highway.
        expanded_graph_.nodes[node_id].clear()
        expanded_graph_.nodes[node_id].update({
            'x': x_val,
            'y': y_val,
            'type_encoded': type_encoded,
            'highway': highway_val  # Preserve the highway attribute for candidate filtering.
        })

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


def normalize_features(grid_features):
    scalers = {
        'density_rank': MinMaxScaler(feature_range=(0, 1)).fit([[1], [5]]),
        'poi_score': MinMaxScaler(feature_range=(0, 1)).fit([[1], [2000]]),
        'temp': MinMaxScaler(feature_range=(0, 1)).fit([[-10], [35]]),
        'rain': lambda x: x
    }

    density_norm = scalers['density_rank'].transform([[grid_features['density_rank']]])[0, 0]
    poi_norm = scalers['poi_score'].transform([[grid_features['poi_score']]])[0, 0]
    temp_norm = scalers['temp'].transform([[grid_features['temp']]])[0, 0]
    rain_norm = scalers['rain'](grid_features['rain'])

    return torch.tensor([density_norm, poi_norm, temp_norm, rain_norm], dtype=torch.float)


def calculate_stop_score(grid_features):
    """Calculate stop likelihood score using urban factors"""
    # Normalize features manually
    density_norm = grid_features['density_rank'] / 5.0  # Assuming 1-5 scale
    poi_norm = grid_features['poi_score'] / 1000.0  # Scale POI scores
    temp_norm = (grid_features['temp'] + 10) / 45.0  # Normalize temp between -10°C to 35°C
    rain_norm = grid_features['rain']  # Binary 0/1

    # Weighted combination (adjust weights as needed)
    score = (0.4 * density_norm +
             0.3 * poi_norm +
             0.2 * temp_norm -
             0.1 * rain_norm)

    return score


def predict_stops(grid_data, road_graph):
    """Predict stops based on urban factors without neural network"""
    base_score = calculate_stop_score(grid_data)

    node_list = sorted(road_graph.nodes())
    candidates = []

    for node in node_list:
        node_attrs = road_graph.nodes[node]

        # Filter by allowed road types
        if node_attrs.get('highway', '') not in Config.ALLOWED_HIGHWAYS:
            continue

        # Filter out junctions
        if road_graph.out_degree(node) > 2:
            continue

        # Add small randomness to distribute stops
        final_score = base_score * (0.9 + 0.2 * np.random.random())

        if final_score > Config.PREDICTION_THRESHOLD:
            candidates.append({
                'lat': node_attrs['y'],
                'lon': node_attrs['x'],
                'score': final_score
            })

    return candidates


def save_predictions(predictions, filename):
    df = pd.DataFrame(predictions)
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


def load_pretrained_model(path):
    model = BusStopPredictor()
    pretrained_dict = torch.load(path, weights_only=True)

    # Handle DataParallel prefixes if present
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    # Key mapping for SAGEConv components
    key_mapping = {
        'conv1.lin.weight': 'conv1.lin_l.weight',
        'conv1.bias': 'conv1.lin_l.bias',
        'conv2.lin.weight': 'conv2.lin_l.weight',
        'conv2.bias': 'conv2.lin_l.bias',
        # Add batch norm mappings if needed
        'bn1.weight': 'bn1.module.weight',
        'bn1.bias': 'bn1.module.bias'
    }

    aligned_dict = {}
    for key in model.state_dict():
        src_key = key_mapping.get(key, key)
        if src_key in pretrained_dict:
            aligned_dict[key] = pretrained_dict[src_key]
        else:
            aligned_dict[key] = model.state_dict()[key]  # Initialize missing params

    model.load_state_dict(aligned_dict, strict=False)
    return model


def create_map(all_predictions, city_grid_data, city_center):
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df = predictions_df.dropna(subset=['lat', 'lon'])
    predictions_df = predictions_df.rename(columns={'lat': 'Latitude', 'lon': 'Longitude'})

    predictions_df[['Latitude', 'Longitude']] = predictions_df[['Latitude', 'Longitude']].apply(pd.to_numeric,
                                                                                                errors='coerce')

    avg_lat, avg_lon = city_center

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14, tiles='CartoDB positron')

    grid_layer = folium.FeatureGroup(name='Density Grid')
    for _, grid in city_grid_data.iterrows():
        if all(col in grid for col in ['min_lat', 'min_lon', 'max_lat', 'max_lon', 'density_rank']):
            grid_layer.add_child(
                folium.Rectangle(
                    bounds=[[grid['min_lat'], grid['min_lon']],
                            [grid['max_lat'], grid['max_lon']]],
                    color='#ff0000',
                    fill=True,
                    fill_color='YlOrRd',
                    fill_opacity=0.2 * grid['density_rank'],
                    popup=f"Density Rank: {grid['density_rank']}"
                )
            )
    m.add_child(grid_layer)

    pred_layer = folium.FeatureGroup(name='Predicted Stops')
    for _, pred in predictions_df.iterrows():
        pred_layer.add_child(
            folium.CircleMarker(
                location=[pred['Latitude'], pred['Longitude']],
                radius=5 + 8 * pred['score'],
                color='#0066cc',
                fill=True,
                fill_opacity=0.7,
                popup=f"Probability: {pred['score']:.2f}<br>"
            )
        )
    m.add_child(pred_layer)

    folium.LayerControl(collapsed=False).add_to(m)
    title_html = '''
         <h3 align="center" style="font-size:16px"><b>Bus Stop Predictions</b></h3>
         <div style="text-align:center;">
             <span style="color: #00cc00;">■</span> Existing Stops &nbsp;
             <span style="color: #0066cc;">■</span> Predicted Stops
         </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    return m


def plot_predictions(city_grid_data, predictions):
    """
    Create a lightweight static map using Matplotlib.
    Plots grid boundaries, existing stops, and predicted stops.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot grid boundaries as rectangles.
    for _, grid in city_grid_data.iterrows():
        min_lon = grid['min_lon']
        min_lat = grid['min_lat']
        max_lon = grid['max_lon']
        max_lat = grid['max_lat']
        width = max_lon - min_lon
        height = max_lat - min_lat
        rect = plt.Rectangle((min_lon, min_lat), width, height,
                             linewidth=1, edgecolor='red', facecolor='none', alpha=0.5)
        ax.add_patch(rect)

    # Convert predictions to a DataFrame and plot.
    pred_df = pd.DataFrame(predictions)
    if not pred_df.empty:
        ax.scatter(pred_df['lon'], pred_df['lat'], color='blue', label='Predicted Stops',
                   alpha=0.7, s=50)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.set_title("Bus Stop Predictions (Matplotlib)")
    plt.savefig("predictions.png")
    plt.show()


def main():
    city_grid_data = read_city_grid(GRID_FILE)
    poi_names, poi_ranks = read_poi_tags(POI_TAGS_FILE)
    tag_rank_mapping = dict(zip(poi_names, poi_ranks))
    temperature, is_raining = get_weather(CITY_NAME, DATE_TIME)

    road_ = download_road_network(CITY_NAME)

    # Load pretrained model
    model = load_pretrained_model(MODEL_SAVE_PATH)
    model.eval()

    city_center = [city_grid_data['min_lat'].mean(), city_grid_data['min_lon'].mean()]

    all_predictions = []
    for _, grid in city_grid_data.iterrows():
        pois, poi_count = get_pois(
            grid["min_lat"], grid["min_lon"],
            grid["max_lat"], grid["max_lon"],
            'amenity', tag_rank_mapping=tag_rank_mapping
        )
        grid_features = extract_grid_features(grid, poi_count, temperature, is_raining)

        candidates = predict_stops(grid_features, road_)

        # Add grid info to predictions
        for candidate in candidates:
            candidate.update({
                'grid_min_lat': grid['min_lat'],
                'grid_max_lat': grid['max_lat'],
                'grid_min_lon': grid['min_lon'],
                'grid_max_lon': grid['max_lon']
            })
        all_predictions.extend(candidates)

    if temperature is not None:
        print(f"\nWeather in {CITY_NAME} on {DATE_TIME}:")
        print(f"Temperature: {temperature}°C")
        print(f"Raining: {'Yes' if is_raining else 'No'}")
    else:
        print("\nError: Unable to fetch weather for this date (historical data requires a paid plan).")

    # all_predictions = filter_predictions(all_predictions, Config.MIN_STOP_DISTANCE)

    # Save and visualize
    save_predictions(all_predictions, OUTPUT_FILE)

    map_ = create_map(all_predictions, city_grid_data, city_center)
    map_.save("Template/bus_stops_prediction_map.html")

    # Visualize using Matplotlib (static lightweight map)
    # plot_predictions(city_grid_data, all_predictions)

    print("Prediction complete.")


if __name__ == "__main__":
    main()

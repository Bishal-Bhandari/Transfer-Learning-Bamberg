import datetime
import pandas as pd
import json
import requests
import torch
import math
import numpy as np
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
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

# OSMnx settings
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.cache_folder = "osmnx_cache_bamberg"
tqdm.pandas()

# Logging settings
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
DATE_TIME = "2025-03-26 09:00"  # Valid date within 5 days
GRID_FILE = "Training Data/city_grid_density_bamberg.ods"
STOPS_FILE = "Training Data/stib_stops.ods"
POI_TAGS_FILE = "poi_tags.json"
MODEL_SAVE_PATH = "Output/best_bus_stop_model.pth"
OUTPUT_FILE = "Model Data/bus_stop_predictions.csv"
DEFAULT_TEMP = 15
DEFAULT_RAIN = False
JUNCTION_BUFFER = 50  # m
CELL_SIZE = 500  # m

class Config:
    DENSITY_MAP ={5: 1, 4: 0.8, 3: 0.7, 2: 0.2, 1: 0.1}
    CITY_NAME = "Bamberg"
    MIN_STOP_DISTANCE = 200  # m
    PREDICTION_THRESHOLD = 0.65
    RADIUS_ROAD_NETWORK = 0
    ROAD_TYPES = ['motorway', 'trunk', 'primary', 'secondary']




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



def read_city_grid(file_path):
    df = pd.read_excel(file_path, engine="odf")
    df = df[["min_lat",	"max_lat",	"min_lon", "max_lon", "density_rank"]].dropna()
    df['density_rank'] = df['density_rank'].astype(int)
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

        # for idx, poi in enumerate(pois[:5], 1):
        #     print(f"{idx}. {poi['name']}")
        #     print(f"   Coordinates: ({poi['latitude']}, {poi['longitude']})")
        #     print(f"   {poi_type.capitalize()}: {poi['tags'].get(poi_type, 'N/A')}\n")
        return pois, popularity_total

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return [], 0


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def filter_by_grid_density_and_time(predictions, date_time_str):

    # Time fraction based time.
    dt = datetime.datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")
    hour = dt.hour
    if 6 <= hour < 10:
        time_fraction = 1
    elif 10 <= hour < 16:
        time_fraction = 0.95
    elif 16 <= hour < 18:
        time_fraction = 0.9
    elif 18 <= hour < 22:
        time_fraction = 0.8
    else:
        time_fraction = 0.5

    grid_groups = {}
    for p in predictions:
        grid_key = (
            p.get('grid_min_lat'),
            p.get('grid_min_lon'),
            p.get('grid_max_lat'),
            p.get('grid_max_lon')
        )
        grid_groups.setdefault(grid_key, []).append(p)

    filtered_predictions = []
    for grid_key, group in grid_groups.items():
        density = int(group[0].get('density_rank', 3))
        base_keep = Config.DENSITY_MAP.get(density, 0.6)
        effective_keep_fraction = 0.9 * base_keep + 0.1 * time_fraction

        sorted_group = sorted(group, key=lambda x: x['score'], reverse=True)
        num_to_keep = max(1, int(round(effective_keep_fraction * len(sorted_group))))

        filtered_predictions.extend(sorted_group[:num_to_keep])

    return filtered_predictions


def filter_predicted_stops(predictions, date_time_str):

    # Filter based on grid.
    filtered_predictions = filter_by_grid_density_and_time(predictions, date_time_str)

    final_candidates = []

    sorted_candidates = sorted(filtered_predictions, key=lambda x: x['score'], reverse=True)

    for candidate in sorted_candidates:
        lat, lon = candidate['lat'], candidate['lon']
        too_close = False
        for selected in final_candidates:
            sel_lat, sel_lon = selected['lat'], selected['lon']
            if haversine_distance(lat, lon, sel_lat, sel_lon) < Config.MIN_STOP_DISTANCE:
                too_close = True
                break
        if not too_close:
            final_candidates.append(candidate)

    return final_candidates


def extract_grid_features(grid, pois_count, temperature, is_raining):

    # Get density rank from grid.
    density_rank = int(grid.get("density_rank", 3))

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

    grid_features = {
        "grid_data": grid_data,
        "poi_score": pois_count,
        "density_rank": density_rank,
        "temp": temperature,
        "rain": rain_val
    }

    return grid_features


def _road_type_to_numeric(road_type):
    if isinstance(road_type, list):
        road_type = road_type[0]

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

    expanded_graph_ = ox.graph_from_place(place_name, network_type='drive', simplify=False)

    # # Get the bounding box of the city
    # nodes = ox.graph_to_gdfs(expanded_graph_, nodes=True, edges=False)
    # west, south, east, north = nodes.union_all().bounds
    # print(west, south, east, north)
    # radius = Config.RADIUS_ROAD_NETWORK
    #
    # # Expand the bounding box
    # expanded_north = north + radius
    # expanded_south = south - radius
    # expanded_east = east + radius
    # expanded_west = west - radius
    #
    # # Download the graph
    # expanded_graph_ = ox.graph_from_bbox(
    #     (expanded_north, expanded_south, expanded_east, expanded_west),
    #     network_type='drive',
    #     simplify=True
    # )

    # Update each node.
    for node_id in list(expanded_graph_.nodes()):
        attrs = expanded_graph_.nodes[node_id]

        # Check if coordinates exist.
        x_val = attrs.get('x')
        y_val = attrs.get('y')
        if x_val is None or y_val is None:
            x_val = attrs.get('lon', 0.0)
            y_val = attrs.get('lat', 0.0)

        # Highway attribute
        highway_val = attrs.get('highway', 'unclassified')
        type_encoded = _road_type_to_numeric(highway_val)

        # Update with required ones
        expanded_graph_.nodes[node_id].clear()
        expanded_graph_.nodes[node_id].update({
            'x': x_val,
            'y': y_val,
            'type_encoded': type_encoded,
            'highway': highway_val
        })

    return expanded_graph_


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


def predict_stops(model, grid_data, road_graph):

    if grid_data['density_rank'] <= 2:
        threshold = 0.4
    elif grid_data['density_rank'] == 3:
        threshold = 0.55
    else:
        threshold = 0.6


    model.eval()
    with torch.no_grad():
        # Clear edge attributes to ensure consistency.
        for u, v, k in road_graph.edges(keys=True):
            road_graph.edges[u, v, k].clear()

        pyg_data = from_networkx(
            road_graph,
            group_node_attrs=['x', 'y', 'type_encoded']
        )
        # Normalize the grid features
        pyg_data.x = normalize_features(grid_data).repeat(pyg_data.num_nodes, 1)
        # Run the model to get predictions.
        pred = model(pyg_data.x, pyg_data.edge_index)
    # Sorted list of node.
    node_list = sorted(road_graph.nodes())

    # Allowed highway types.
    allowed_highways = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified']

    candidates = []
    for idx, node in enumerate(node_list):
        node_attrs = road_graph.nodes[node]

        highway_val = node_attrs.get('highway', 'unclassified')
        if highway_val not in allowed_highways:
            continue

        # Check total degree for junction.
        if road_graph.degree(node) <= 1:
            continue

        point = Point(node_attrs['x'], node_attrs['y'])

        node_buffer = point.buffer(JUNCTION_BUFFER / 111000)

        # Check if the node is a junction.
        is_junction = (road_graph.out_degree(node) > 2)

        if not is_junction and pred[idx] > threshold:
            candidates.append({
                'lat': node_attrs['y'],
                'lon': node_attrs['x'],
                'score': pred[idx].item()
            })
    return candidates


def save_predictions(predictions, filename):
    df = pd.DataFrame(predictions)
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


def load_pretrained_model(path):
    model = BusStopPredictor()
    pretrained_dict = torch.load(path, weights_only=True)

    # Check for DataParallel prefixes
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    # Mapping for SAGEConv components
    key_mapping = {
        'conv1.lin.weight': 'conv1.lin_l.weight',
        'conv1.bias': 'conv1.lin_l.bias',
        'conv2.lin.weight': 'conv2.lin_l.weight',
        'conv2.bias': 'conv2.lin_l.bias',

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


def create_map(all_predictions, city_center, city_grid_data):

    predictions_df = pd.DataFrame(all_predictions)
    predictions = predictions_df.dropna(subset=['lat', 'lon'])

    avg_lat, avg_lon = city_center

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12, tiles="OpenStreetMap")
    # For grid
    grid_layer = folium.FeatureGroup(name='Density Grid')
    for _, grid in city_grid_data.iterrows():
        if all(col in grid for col in ['min_lat', 'min_lon', 'max_lat', 'max_lon', 'density_rank']):
            grid_layer.add_child(
                folium.Rectangle(
                    bounds=[[grid['min_lat'], grid['min_lon']],
                            [grid['max_lat'], grid['max_lon']]],
                    color='#00ff00',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.2 * grid['density_rank']
                )
            )
    m.add_child(grid_layer)
    #For bus stop nodes
    pred_layer = folium.FeatureGroup(name='Predicted Stops')
    for _, pred in predictions.iterrows():
        pred_layer.add_child(
            folium.CircleMarker(
                location=[pred['lat'], pred['lon']],
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
             <span style="color: #0066cc;">■</span> Predicted Stops
         </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    return m


def plot_predictions(city_grid_data, predictions):

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot grid.
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

    # To a DataFrame and plot.
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
    stib_stops_data = read_stib_stops(STOPS_FILE)
    poi_names, poi_ranks = read_poi_tags(POI_TAGS_FILE)
    tag_rank_mapping = dict(zip(poi_names, poi_ranks))
    temperature, is_raining = get_weather(CITY_NAME, DATE_TIME)

    road_ = download_road_network(CITY_NAME)

    # Load pretrained model
    model = load_pretrained_model(MODEL_SAVE_PATH)
    model.eval()

    # Get city center for map
    city_center = [city_grid_data['min_lat'].mean(), city_grid_data['min_lon'].mean()]

    all_predictions = []
    for _, grid in city_grid_data.iterrows():
        pois, poi_count = get_pois(
            grid["min_lat"], grid["min_lon"],
            grid["max_lat"], grid["max_lon"],
            'amenity', tag_rank_mapping=tag_rank_mapping
        )
        grid_features = extract_grid_features(grid, poi_count, temperature, is_raining)

        grid_boundary = box(grid["min_lon"], grid["min_lat"], grid["max_lon"], grid["max_lat"])
        grid_nodes = []

        for node, data in road_.nodes(data=True):
            if grid_boundary.contains(Point(data['x'], data['y'])):
                grid_nodes.append(node)

        # Skip grids with no roads
        if len(grid_nodes) == 0:
            continue

        grid_road_network = road_.subgraph(grid_nodes).copy()

        has_required_attrs = all(
            all(attr in data for attr in ['x', 'y', 'type_encoded'])
            for _, data in grid_road_network.nodes(data=True)
        )

        if not has_required_attrs:
            continue

        # Grid specific network for prediction
        try:
            candidates = predict_stops(model, grid_features, grid_road_network)

            # Add grid info to predictions
            for candidate in candidates:
                candidate.update({
                    'grid_min_lat': grid['min_lat'],
                    'grid_max_lat': grid['max_lat'],
                    'grid_min_lon': grid['min_lon'],
                    'grid_max_lon': grid['max_lon'],
                    'density_rank': grid['density_rank']
                })

            all_predictions.extend(candidates)
        except Exception as e:
            continue


    if temperature is not None:
        print(f"\nWeather in {CITY_NAME} on {DATE_TIME}:")
        print(f"Temperature: {temperature}°C")
        print(f"Raining: {'Yes' if is_raining else 'No'}")
    else:
        print("\nError: Unable to get weather for this date (historical data requires a paid plan).")


    all_predictions = filter_predicted_stops(all_predictions, DATE_TIME)

    # Save and visualize
    save_predictions(all_predictions, OUTPUT_FILE)

    map_ = create_map(all_predictions, city_center, city_grid_data)
    map_.save("Template/bus_stops_prediction_map_bamberg1.html")

    # plot_predictions(city_grid_data, all_predictions)

    print("Prediction complete.")

if __name__ == "__main__":
    main()
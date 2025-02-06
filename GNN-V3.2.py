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
import networkx as nx

with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)

# API key
API_KEY = api_keys['Weather_API']['API_key']

# User input
# Example usage
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



def process_grid_roads(grid):
    """Process road network for a grid cell with feature engineering"""

    # Correct bounding box order (left, bottom, right, top)
    bbox = (grid["min_lon"], grid["min_lat"], grid["max_lon"], grid["max_lat"])

    # Fetch road network
    G = ox.graph_from_bbox(bbox, network_type='drive', simplify=True)

    # Ensure edges have geometries before adding bearings
    for u, v, data in G.edges(data=True):
        if 'geometry' not in data:
            # Approximate geometry as a straight line if missing
            node_u, node_v = G.nodes[u], G.nodes[v]
            data['geometry'] = ox.utils_geo.LineString([(node_u['x'], node_u['y']), (node_v['x'], node_v['y'])])

    # Custom bearing calculation to replace outdated ox.add_edge_bearings
    for u, v, data in G.edges(data=True):
        geom = data['geometry']
        coords = list(geom.coords)
        if len(coords) < 2:
            continue  # Shouldn't happen as we added LineString with 2 points
        start = coords[0]
        end = coords[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        radians = np.arctan2(dy, dx)
        bearing = np.degrees(radians) % 360
        data['bearing'] = bearing

    # Feature engineering
    for u, v, data in G.edges(data=True):
        data.setdefault("highway", "")
        data['road_type'] = int(data["highway"] in {'motorway', 'trunk'})

        # Compute curvature (absolute bearing difference for u-v and v-u directions)
        bearing = data.get('bearing', 0)
        reverse_bearing = (bearing + 180) % 360  # Approximate reverse direction
        data['curvature'] = np.abs(bearing - reverse_bearing)

        # Identify junctions
        data['is_junction'] = (G.degree(u) > 2 or G.degree(v) > 2)

    return G


def create_gnn_data(G, grid_features):
    # Node features: [latitude, longitude, degree]
    nodes = []
    node_id_to_index = {}  # Mapping from node ID to index
    for idx, node in enumerate(G.nodes()):
        n_data = G.nodes[node]
        nodes.append([n_data['y'], n_data['x'], G.degree(node)])
        node_id_to_index[node] = idx

    # Edge features: [road_type, length, curvature, is_junction]
    edges = []
    edge_indices = []
    for u, v, data in G.edges(data=True):
        edges.append([
            data.get('road_type', 0),
            data.get('length', 0),
            data.get('curvature', 0),
            int(data.get('is_junction', 0))
        ])
        edge_indices.append([u, v])

    # Convert grid features to tensors
    grid_data_values = [
        grid_features['grid_data']['min_lat'],
        grid_features['grid_data']['max_lat'],
        grid_features['grid_data']['min_lon'],
        grid_features['grid_data']['max_lon'],
        grid_features['grid_data']['density_rank']
    ]
    grid_data = torch.tensor(grid_data_values, dtype=torch.float32)
    poi_score = torch.tensor([grid_features['poi_score']], dtype=torch.float32)
    density_rank = torch.tensor([grid_features['density_rank']], dtype=torch.float32)
    temperature = torch.tensor([grid_features['temp']], dtype=torch.float32)
    # Convert Boolean to float: 1.0 if raining, 0.0 otherwise.
    rain = torch.tensor([1.0 if grid_features['rain'] else 0.0], dtype=torch.float32)

    grid_feature_tensor = torch.cat([grid_data, poi_score, density_rank, temperature, rain], dim=-1)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4 + grid_feature_tensor.numel()))
    else:
        edge_features = [
            torch.cat([torch.tensor(e, dtype=torch.float32), grid_feature_tensor])
            for e in edges
        ]
        edge_attr = torch.stack(edge_features)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    data_obj = Data(
        x=torch.tensor(nodes, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    # Store the mapping in the Data object for later use
    data_obj.node_id_to_index = node_id_to_index
    return data_obj



class BusStopGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.predictor = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return torch.sigmoid(self.predictor(x))


def train_model(grids_data, stib_stops):
    model = BusStopGNN(num_features=3)  # Input node features
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        for grid in grids_data:
            # Create graph data with POI, weather, and density features
            grid_features = {
                'poi_score': grid['poi_score'],
                'density': grid['density_rank'],
                'temperature': grid['temp'],
                'rain': grid['rain']
            }
            data = create_gnn_data(grid['graph'], grid_features)

            # Generate labels (1 if near existing stop)
            labels = torch.zeros(data.x.size(0))
            for stop in stib_stops:
                # Find nearest node to stop coordinates
                nearest_node = ox.distance.nearest_nodes(
                    grid['graph'],
                    X=stop['lon'],
                    Y=stop['lat']
                )
                labels[nearest_node] = 1

            # Train
            pred = model(data)
            loss = f.binary_cross_entropy(pred.squeeze(), labels)
            loss.backward()
            optimizer.step()


def predict_new_stops(model, grid):
    """Generate predictions for new bus stops in a grid"""
    data = create_gnn_data(grid['graph'], grid['features'])
    with torch.no_grad():
        probas = model(data)

    # Filter candidates
    candidates = []
    for node_idx, prob in enumerate(probas):
        node_data = grid['graph'].nodes[node_idx]
        edge_data = list(grid['graph'].edges(node_idx, data=True))[0][2]

        # Exclusion criteria
        if (edge_data['curvature'] < 45 and
                not edge_data['is_junction'] and
                prob > 0.7):
            candidates.append({
                'lat': node_data['y'],
                'lon': node_data['x'],
                'probability': prob.item()
            })

    return sorted(candidates, key=lambda x: -x['probability'])

def main():
    city_grid_data = read_city_grid(city_grid_file)
    stib_stops_data = read_stib_stops(stib_stops_file)
    poi_names, poi_ranks = read_poi_tags(poi_tags_file)
    tag_rank_mapping = dict(zip(poi_names, poi_ranks))
    temperature, is_raining = get_weather(city_name, date_time)

    # Initialize the model once
    model = BusStopGNN(num_features=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _, grid in city_grid_data.iterrows():
        pois, poi_count = get_pois(
            grid["min_lat"], grid["min_lon"], grid["max_lat"], grid["max_lon"], 'amenity',
            tag_rank_mapping=tag_rank_mapping
        )
        graph = process_grid_roads(grid)

        current_features = {
            'grid_data': grid,
            'poi_score': poi_count,
            'density_rank': grid['density_rank'],
            'temp': temperature,
            'rain': is_raining
        }

        data = create_gnn_data(graph, current_features)

        # Create labels tensor
        labels = torch.zeros(data.x.size(0))
        node_mapping = data.node_id_to_index  # The mapping from node IDs to tensor indices

        for _, stop in stib_stops_data.iterrows():
            nearest_node = ox.distance.nearest_nodes(
                graph,
                X=stop['stop_lon'],
                Y=stop['stop_lat']
            )
            # Only update if the nearest_node is in the mapping
            if nearest_node in node_mapping:
                labels[node_mapping[nearest_node]] = 1
            else:
                print(f"Warning: Nearest node {nearest_node} not found in node mapping.")

        # Train on this grid's data
        model.train()
        optimizer.zero_grad()
        pred = model(data).squeeze()
        loss = f.binary_cross_entropy(pred, labels)
        loss.backward()
        optimizer.step()

        # Predict new stops in this grid (assuming predict_new_stops uses similar logic)
        model.eval()
        new_stops = predict_new_stops(model, {'graph': graph, 'features': current_features})
        print(f"Predicted {len(new_stops)} new bus stops in grid:")
        for stop in new_stops:
            print(f"  → {stop['lat']:.6f}, {stop['lon']:.6f} (score: {stop['probability']:.2f})")

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



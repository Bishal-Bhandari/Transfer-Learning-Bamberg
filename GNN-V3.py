import json

import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import osmnx as ox
import requests
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from shapely.geometry import Point, box
from overpy import Overpass
from datetime import datetime, time
from sklearn.preprocessing import StandardScaler

# Load the API keys
with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)
# API key
WEATHER_API_KEY = api_keys['Weather_API']['API_key']

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
TIMEZONE = 'Europe/London'  # Adjust based on city

# Mock POI tags and popularity (replace with actual dict)
POI_TAGS = {
    'amenity': {'restaurant': 5, 'school': 4},
    'shop': {'mall': 5, 'supermarket': 4}
}


# Load and clean grid data
def load_grid_data(file_path):
    df = pd.read_excel(file_path, engine='odf')
    df = df.dropna()

    # Split combined coordinate column
    df[['max_lat', 'max_lon']] = df['max_lat_max_lon'].str.split(',', expand=True).astype(float)

    # Validate coordinates
    valid = (
            (df['min_lat'] < df['max_lat']) &
            (df['min_lon'] < df['max_lon'])
    )
    return df[valid].reset_index(drop=True)

# Fetch POIs from OSM
def fetch_pois(grid_bounds):
    overpass = Overpass()
    query = f"""
        [out:json];
        node[{'}][{'}]({grid_bounds['min_lat']},{grid_bounds['min_lon']},{grid_bounds['max_lat']},{grid_bounds['max_lon']});
        out body;
        """
    try:
        result = overpass.query(query)
        return [{'lat': float(node.lat), 'lon': float(node.lon), 'tags': node.tags}
                for node in result.nodes]
    except:
        return []


# Get weather data
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}"
    response = requests.get(url).json()
    temp = response['main']['temp'] - 273.15
    raining = 'rain' in response.get('weather', [{}])[0].get('main', '').lower()
    return raining, temp


# Process time input
def process_time(input_time):
    user_time = datetime.strptime(input_time, "%H:%M").time()
    is_day = time(7, 0) <= user_time <= time(19, 0)
    is_peak = (time(8, 0) <= user_time <= time(10, 0)) or (time(17, 0) <= user_time <= time(19, 0))
    return is_day, is_peak


# Generate candidate points on roads
def generate_candidates(grid_bounds):
    try:
        roads = ox.graph_from_bbox(
            grid_bounds['max_lat'], grid_bounds['min_lat'],
            grid_bounds['max_lon'], grid_bounds['min_lon'],
            network_type='drive'
        )
        nodes, edges = ox.graph_to_gdfs(roads)
        candidates = []
        for _, edge in edges.iterrows():
            line = edge['geometry']
            for dist in np.linspace(0, 1, num=5):
                point = line.interpolate(dist, normalized=True)
                candidates.append((point.y, point.x))
        return candidates
    except:
        return []


# Define GNN model
class EnhancedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(self.lin(x))


# Train and predict new bus stops using GNN
def predict_bus_stops(city, time_input, grid_file, bus_stops_file):
    # Load data with validation
    grids = load_grid_data(grid_file)
    if grids.empty:
        raise ValueError("No valid grids found. Check input file format")

    existing_bus_stops = gpd.read_file(bus_stops_file)
    print(f"Loaded {len(grids)} grids, {len(existing_bus_stops)} existing stops")
    raining, temp = get_weather(city)
    is_day, is_peak = process_time(time_input)

    features = []
    for idx, grid in grids.iterrows():
        pois = fetch_pois(grid)

        # Convert booleans to numerical values
        poi_score = sum(
            POI_TAGS.get(tag, {}).get(value, 1)
            for p in pois
            for tag, value in p.get('tags', {}).items()
        )

        grid_box = box(grid['min_lon'], grid['min_lat'], grid['max_lon'], grid['max_lat'])
        existing_count = existing_bus_stops.geometry.within(grid_box).sum()

        features.append([
            grid['population_density'],
            poi_score,
            int(is_day),  # Convert bool to 0/1
            int(is_peak),
            int(raining),
            temp,
            existing_count
        ])

    if not features:
        raise ValueError("No features generated - check POI fetching and input data")

    X = torch.tensor(StandardScaler().fit_transform(features), dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    model = GNN(in_channels=X.shape[1], hidden_channels=16, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(200):
        optimizer.zero_grad()
        out = model(X, edge_index).squeeze()
        loss = F.mse_loss(out, torch.rand(out.shape))
        loss.backward()
        optimizer.step()

    predictions = (out.detach().numpy() * 3).astype(int)
    new_stops = []

    for idx, grid in grids.iterrows():
        candidates = generate_candidates(grid)
        n_new = predictions[idx]

        for candidate in candidates:
            if n_new > 0:
                new_stops.append({'lat': candidate[0], 'lon': candidate[1]})
                n_new -= 1

    output_df = pd.DataFrame(new_stops)
    output_df.to_excel('new_bus_stops.ods', engine='odf')

    m = folium.Map(location=[grids['min_lat'].mean(), grids['min_lon'].mean()])
    for stop in new_stops:
        folium.Marker([stop['lat'], stop['lon']], icon=folium.Icon(color='green')).add_to(m)


# Example execution
predict_bus_stops(
    city="Brussels",
    time_input="17:30",
    grid_file="Training Data/city_grid_density.ods",
    bus_stops_file="Training Data/stib_stops.ods"
)
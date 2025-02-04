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
import pytz

# Load API keys
with open('api_keys.json') as f:
    api_keys = json.load(f)
WEATHER_API_KEY = api_keys['Weather_API']['API_key']

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
TIMEZONE = 'Europe/Brussels'

# Enhanced POI tags configuration
POI_TAGS = {
    'amenity': {'university': 5, 'hospital': 5, 'restaurant': 4},
    'shop': {'mall': 5, 'supermarket': 4},
    'tourism': {'museum': 5, 'attraction': 4}
}


def load_grid_data(file_path):
    """Load and process grid data with proper validation"""
    df = pd.read_excel(file_path, engine='odf')

    # Split combined coordinates column
    df[['max_lat', 'max_lon']] = df['max_lat_max_lon'].str.split(',', expand=True).astype(float)

    # Validate coordinates
    df = df.dropna()
    valid = (
            (df['min_lat'] < df['max_lat']) &
            (df['min_lon'] < df['max_lon']) &
            (df['population_density'].between(1, 5))
    )
    df = df[valid].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid grids found after filtering")
    return df


def fetch_pois(grid):
    """Fetch POIs from OSM with enhanced query"""
    try:
        overpass = Overpass()
        query = f"""
            [out:json];
            node({grid['min_lat']},{grid['min_lon']},{grid['max_lat']},{grid['max_lon']});
            way({grid['min_lat']},{grid['min_lon']},{grid['max_lat']},{grid['max_lon']});
            relation({grid['min_lat']},{grid['min_lon']},{grid['max_lat']},{grid['max_lon']});
            (._;>;);
            out body;
        """
        result = overpass.query(query)
        return [
            {'lat': float(node.lat), 'lon': float(node.lon), 'tags': node.tags}
            for node in result.nodes if hasattr(node, 'tags')
        ]
    except Exception as e:
        print(f"POI fetch error: {e}")
        return []


class EnhancedGNN(torch.nn.Module):
    """Improved GNN architecture with dropout"""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(self.lin(x))


def create_spatial_edges(grids, threshold=0.02):
    """Create edges between neighboring grids based on spatial proximity"""
    edges = []
    coords = grids[['min_lat', 'min_lon']].values

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if np.linalg.norm(coords[i] - coords[j]) < threshold:
                edges.append([i, j])
                edges.append([j, i])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def predict_bus_stops(city, time_input, grid_file, bus_stops_file):
    # Load and validate data
    grids = load_grid_data(grid_file)
    existing_bus_stops = gpd.read_file(bus_stops_file)

    # Get environmental context
    raining, temp = get_weather(city)
    is_day, is_peak = process_time(time_input)

    # Prepare features
    features = []
    for _, grid in grids.iterrows():
        pois = fetch_pois(grid)

        # Calculate POI score
        poi_score = sum(
            POI_TAGS.get(tag, {}).get(value, 1)
            for p in pois
            for tag, value in p.get('tags', {}).items()
        )

        # Count existing stops
        grid_box = box(grid['min_lon'], grid['min_lat'], grid['max_lon'], grid['max_lat'])
        existing_count = existing_bus_stops.geometry.within(grid_box).sum()

        features.append([
            grid['population_density'],
            poi_score,
            int(is_day),
            int(is_peak),
            int(raining),
            temp,
            existing_count
        ])

    # Convert and scale features
    X = torch.tensor(StandardScaler().fit_transform(features), dtype=torch.float)

    # Create graph structure
    edge_index = create_spatial_edges(grids)

    # Initialize and train GNN
    model = EnhancedGNN(X.shape[1], 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Mock training targets (replace with real labeled data)
    y = torch.tensor(
        np.clip(grids['population_density'] * 0.7 + np.random.rand(len(grids)) * 0.3, 0, 3),
        dtype=torch.float
    )

    # Training loop
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(X, edge_index).squeeze()
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()

    # Generate predictions
    with torch.no_grad():
        predictions = (model(X, edge_index).squeeze().numpy() * 3).astype(int)

    # Generate candidate points
    new_stops = []
    for idx, grid in grids.iterrows():
        candidates = generate_candidates(grid)
        selected = []

        for candidate in candidates:
            candidate_point = Point(candidate[1], candidate[0])

            # Check distance from existing stops
            distances = existing_bus_stops.geometry.distance(candidate_point)
            if distances.min() > 0.0002:  # ~22 meters
                selected.append({'lat': candidate[0], 'lon': candidate[1]})
                if len(selected) >= predictions[idx]:
                    break

        new_stops.extend(selected)

    # Save and visualize results
    output_df = gpd.GeoDataFrame(
        new_stops,
        geometry=[Point(x['lon'], x['lat']) for x in new_stops]
    )
    output_df.to_file('new_bus_stops.geojson', driver='GeoJSON')

    # Create visualization map
    m = folium.Map(location=[grids['min_lat'].mean(), grids['min_lon'].mean()], zoom_start=12)
    for _, row in output_df.iterrows():
        folium.Marker(
            [row.geometry.y, row.geometry.x],
            icon=folium.Icon(color='green', icon='bus', prefix='fa')
        ).add_to(m)
    m.save('bus_stops_prediction.html')


# Helper functions
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}"
        response = requests.get(url).json()
        temp = response['main']['temp'] - 273.15
        raining = any('rain' in w['main'].lower() for w in response.get('weather', []))
        return raining, temp
    except Exception as e:
        print(f"Weather API error: {e}")
        return False, 20.0  # Fallback values


def process_time(input_time):
    try:
        user_time = datetime.strptime(input_time, "%H:%M").time()
        tz = pytz.timezone(TIMEZONE)
        now = datetime.now(tz).replace(
            hour=user_time.hour,
            minute=user_time.minute,
            second=0,
            microsecond=0
        )
        is_day = 7 <= now.hour < 19
        is_peak = (8 <= now.hour < 10) or (17 <= now.hour < 19)
        return is_day, is_peak
    except:
        return True, False


def generate_candidates(grid):
    try:
        roads = ox.graph_from_bbox(
            grid['max_lat'], grid['min_lat'],
            grid['max_lon'], grid['min_lon'],
            network_type='drive'
        )
        edges = ox.graph_to_gdfs(roads, nodes=False)
        candidates = []

        for _, edge in edges.iterrows():
            if 'geometry' in edge:
                for dist in np.linspace(0.1, 0.9, 5):
                    point = edge['geometry'].interpolate(dist, normalized=True)
                    candidates.append((point.y, point.x))

        return candidates
    except Exception as e:
        print(f"Road network error: {e}")
        return []


if __name__ == "__main__":
    predict_bus_stops(
        city="Brussels",
        time_input="17:30",
        grid_file="Training Data/city_grid_density.ods",
        bus_stops_file="Training Data/stib_stops.ods"
    )
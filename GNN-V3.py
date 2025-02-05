import json
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import requests
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from shapely.geometry import Point, box
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pytz
import osmnx as ox
from tqdm import tqdm
import logging
from pathlib import Path
import warnings
from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import Table, TableRow, TableCell
from odf.text import P

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 300

CONFIG = {
    "model_params": {
        "hidden_channels": 32,
        "learning_rate": 0.01,
        "epochs": 200,
        "dropout": 0.5
    },
    "proximity_threshold": 0.00045,  # ~50 meters
    "max_candidates_per_grid": 5,
    "output_dir": "output",
    "junction_threshold": 0.00018,  # ~20 meters
    "poi_tags": {
        "amenity": ["hospital", "school", "university", "bus_station"],
        "shop": ["supermarket", "mall"],
        "public_transport": ["station"]
    }
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)


class EnhancedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(self.lin(x))


class DataHandler:
    @staticmethod
    def load_grid_data(file_path):
        df = pd.read_excel(file_path, engine='odf')
        required_columns = ['min_lat', 'min_lon', 'max_lat', 'max_lon', 'population_density']

        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required grid columns")

        # Coordinate validation
        df['min_lat'] = df['min_lat'].clip(-90, 90)
        df['max_lat'] = df['max_lat'].clip(-90, 90)
        df['min_lon'] = df['min_lon'].clip(-180, 180)
        df['max_lon'] = df['max_lon'].clip(-180, 180)

        # Ensure proper coordinate order
        df[['min_lat', 'max_lat']] = np.sort(df[['min_lat', 'max_lat']], axis=1)
        df[['min_lon', 'max_lon']] = np.sort(df[['min_lon', 'max_lon']], axis=1)

        return df[df['population_density'].between(1, 5)].reset_index(drop=True)


class POIAnalyzer:
    def __init__(self):
        self.osm_tags = self._process_config(CONFIG["poi_tags"])
        self.poi_types = list({item for sublist in CONFIG["poi_tags"].values() for item in sublist})

    def _process_config(self, config):
        return {category: list(values) for category, values in config.items()}

    def get_poi_counts(self, grid):
        try:
            bbox = (grid['max_lat'], grid['min_lat'], grid['max_lon'], grid['min_lon'])
            pois = ox.features.features_from_bbox(*bbox, tags=self.osm_tags)
            return self._count_pois(pois)
        except Exception as e:
            logger.error(f"POI error: {str(e)}")
            return {pt: 0 for pt in self.poi_types}

    def _count_pois(self, pois_gdf):
        counts = {pt: 0 for pt in self.poi_types}
        for col in pois_gdf.columns:
            for pt in self.poi_types:
                if col.endswith(pt):
                    counts[pt] += len(pois_gdf[~pois_gdf[col].isna()])
        return counts


class RoadNetwork:
    @staticmethod
    def generate_candidates(grid):
        try:
            roads = ox.graph_from_bbox(
                grid['max_lat'], grid['min_lat'],
                grid['max_lon'], grid['min_lon'],
                network_type='drive', simplify=False
            )
            nodes, edges = ox.graph_to_gdfs(roads)
            candidates = []

            for _, edge in edges.iterrows():
                if not hasattr(edge.geometry, 'interpolate'):
                    continue

                u_point = nodes.loc[edge['u'], 'geometry']
                v_point = nodes.loc[edge['v'], 'geometry']

                for dist in np.linspace(0.2, 0.8, 5):
                    point = edge.geometry.interpolate(dist, normalized=True)
                    if point.distance(u_point) < CONFIG["junction_threshold"]:
                        continue
                    if point.distance(v_point) < CONFIG["junction_threshold"]:
                        continue

                    candidates.append({
                        'lat': point.y,
                        'lon': point.x,
                        'grid_id': grid.name
                    })

            return candidates[:CONFIG["max_candidates_per_grid"]]
        except Exception as e:
            logger.error(f"Candidate error: {str(e)}")
            return []


class ModelTrainer:
    def __init__(self, input_size):
        self.model = EnhancedGNN(input_size,
                                 CONFIG["model_params"]["hidden_channels"],
                                 CONFIG["model_params"]["dropout"])
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=CONFIG["model_params"]["learning_rate"])
        self.epochs = CONFIG["model_params"]["epochs"]

    def train(self, X, edge_index, y):
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training"):
            self.optimizer.zero_grad()
            pred = self.model(X, edge_index).squeeze()
            loss = F.mse_loss(pred, y)
            loss.backward()
            self.optimizer.step()
        return self.model


def create_spatial_graph(grids):
    centroids = grids.apply(lambda r: [
        (r['min_lat'] + r['max_lat']) / 2,
        (r['min_lon'] + r['max_lon']) / 2
    ], axis=1).tolist()

    edges = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            if np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j])) < 0.02:
                edges.extend([[i, j], [j, i]])

    return torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)


def get_weather():
    try:
        with open('api_keys.json') as f:
            api_key = json.load(f)['openweathermap']

        url = f"http://api.openweathermap.org/data/2.5/weather?q=Brussels&appid={api_key}"
        response = requests.get(url).json()
        temp = response['main']['temp'] - 273.15  # Convert Kelvin to Celsius
        raining = any('rain' in w['main'].lower() for w in response.get('weather', []))
        return [int(raining), temp]
    except Exception as e:
        logger.error(f"Weather error: {str(e)}")
        return [0, 20.0]


def get_time_features():
    try:
        tz = pytz.timezone('Europe/Brussels')
        now = datetime.now(tz)
        return [
            int(7 <= now.hour < 19),
            int((8 <= now.hour < 10) or (17 <= now.hour < 19))
        ]
    except Exception as e:
        logger.error(f"Time error: {str(e)}")
        return [1, 0]


def validate_candidates(candidates, existing_stops):
    valid = []
    gdf = gpd.GeoDataFrame(
        geometry=[Point(c['lon'], c['lat']) for c in candidates],
        crs="EPSG:4326"
    )

    if not existing_stops.empty:
        distances = gdf.geometry.apply(
            lambda x: existing_stops.geometry.distance(x).min()
        )
        valid = [c for c, d in zip(candidates, distances) if d > CONFIG["proximity_threshold"]]
    else:
        valid = candidates

    return valid


def save_results(results_df, grids):
    # Save to ODS
    doc = OpenDocumentSpreadsheet()
    table = Table(name="BusStops")

    # Header
    tr = TableRow()
    for col in results_df.columns:
        tc = TableCell()
        tc.addElement(P(text=col))
        tr.addElement(tc)
    table.addElement(tr)

    # Data
    for _, row in results_df.iterrows():
        tr = TableRow()
        for val in row:
            tc = TableCell()
            tc.addElement(P(text=str(val)))
            tr.addElement(tc)
        table.addElement(tr)

    doc.spreadsheet.addElement(table)
    doc.save(Path(CONFIG["output_dir"]) / "bus_stops.ods")

    # Create map
    m = folium.Map(
        location=[results_df['lat'].mean(), results_df['lon'].mean()],
        zoom_start=13,
        tiles='CartoDB positron'
    )

    for _, row in results_df.iterrows():
        folium.Marker(
            [row['lat'], row['lon']],
            popup=f"Grid {row['grid_id']}",
            icon=folium.Icon(color='blue', icon='bus', prefix='fa')
        ).add_to(m)

    for _, grid in grids.iterrows():
        folium.Rectangle(
            bounds=[[grid['min_lat'], grid['min_lon']], [grid['max_lat'], grid['max_lon']]],
            color='#ff7800',
            fill=True,
            fill_opacity=0.2
        ).add_to(m)

    m.save(Path(CONFIG["output_dir"]) / "bus_stops_map.html")


def main():
    try:
        # Load data
        grids = DataHandler.load_grid_data("Training Data/city_grid_density.ods")
        existing_stops = gpd.read_file("Training Data/stib_stops.ods")
        existing_stops = gpd.GeoDataFrame(
            existing_stops,
            geometry=gpd.points_from_xy(existing_stops['stop_lon'], existing_stops['stop_lat']),
            crs="EPSG:4326"
        )

        # Feature engineering
        poi_analyzer = POIAnalyzer()
        weather = get_weather()
        time_features = get_time_features()
        features = []

        logger.info("Processing grids...")
        for idx, grid in tqdm(grids.iterrows(), total=len(grids)):
            grid_poly = box(grid['min_lon'], grid['min_lat'], grid['max_lon'], grid['max_lat'])
            poi_counts = poi_analyzer.get_poi_counts(grid)
            existing_count = existing_stops[existing_stops.within(grid_poly)].shape[0]

            features.append([
                grid['population_density'],
                *[poi_counts[pt] for pt in poi_analyzer.poi_types],
                *time_features,
                *weather,
                existing_count
            ])

        # Prepare model data
        X = torch.tensor(StandardScaler().fit_transform(features), dtype=torch.float)
        edge_index = create_spatial_graph(grids)
        y = torch.tensor(grids['population_density'].values * 0.5 + np.random.rand(len(grids)) * 0.5,
                         dtype=torch.float)

        # Train model
        trainer = ModelTrainer(X.shape[1])
        model = trainer.train(X, edge_index, y)

        # Generate predictions
        with torch.no_grad():
            predictions = model(X, edge_index).squeeze().numpy() * 3
        predictions = np.clip(predictions, 0, 3).astype(int)

        # Generate candidates
        results = []
        logger.info("Generating candidates...")
        for idx, grid in tqdm(grids.iterrows(), total=len(grids)):
            candidates = RoadNetwork.generate_candidates(grid)
            valid_candidates = validate_candidates(candidates, existing_stops)
            results.extend(valid_candidates[:predictions[idx]])

        # Save results
        if results:
            results_df = pd.DataFrame(results)
            save_results(results_df, grids)
            logger.info(f"Saved {len(results_df)} bus stop recommendations")
        else:
            logger.warning("No valid bus stops generated")

    except Exception as e:
        logger.error(f"Main error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
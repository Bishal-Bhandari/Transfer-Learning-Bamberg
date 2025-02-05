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
from typing import Dict, List, Tuple
from pathlib import Path
import warnings
from haversine import haversine

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OSMnx configuration
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 300

# Configuration constants
CONFIG = {
    "model_params": {
        "hidden_channels": 32,
        "learning_rate": 0.01,
        "epochs": 500,
        "dropout": 0.5
    },
    "proximity_threshold_meters": 50,
    "junction_threshold_meters": 10,
    "max_candidates_per_grid": 5,
    "output_dir": "output",
    "default_poi_weights": {
        "amenity": {"hospital": 5, "school": 4},
        "shop": {"supermarket": 3}
    }
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)


class EnhancedGNN(torch.nn.Module):
    """Graph Neural Network for bus stop prediction"""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(self.lin(x))


class DataLoader:
    """Handles data loading and preprocessing"""

    @staticmethod
    def load_grid_data(file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(file_path, engine='odf')
            df = df.rename(columns={'density_rank': 'population_density'})
            required_columns = ['min_lat', 'min_lon', 'max_lat', 'max_lon', 'population_density']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")

            # Coordinate validation and normalization
            for col in ['min_lat', 'max_lat']:
                df[col] = df[col].clip(-90, 90)
            for col in ['min_lon', 'max_lon']:
                df[col] = df[col].clip(-180, 180)

            df = df[(df['population_density'].between(1, 5))].reset_index(drop=True)
            if df.empty:
                raise ValueError("No valid grids after processing")
            logger.info(f"Loaded {len(df)} valid grids")
            return df
        except Exception as e:
            logger.error(f"Error loading grid data: {str(e)}")
            raise

    @staticmethod
    def load_poi_config(file_path: str) -> Dict:
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Using default POI config: {str(e)}")
            return CONFIG["default_poi_weights"]


class POIProcessor:
    """Handles POI data processing"""

    def __init__(self, poi_config: Dict):
        self.poi_config = poi_config

    def fetch_pois(self, grid: Dict) -> List[Dict]:
        try:
            pois_gdf = ox.features.features_from_bbox(
                grid['max_lat'], grid['min_lat'],
                grid['max_lon'], grid['min_lon'],
                tags=self.poi_config
            )
            return self._process_pois(pois_gdf)
        except Exception as e:
            logger.error(f"POI fetch error: {str(e)}")
            return []

    def _process_pois(self, pois_gdf: gpd.GeoDataFrame) -> List[Dict]:
        pois = []
        for _, row in pois_gdf.iterrows():
            try:
                geom = row.geometry
                if geom.geom_type not in ['Point', 'Polygon']:
                    continue
                centroid = geom.centroid if geom.geom_type == 'Polygon' else geom
                pois.append({'lat': centroid.y, 'lon': centroid.x})
            except Exception as e:
                logger.warning(f"Error processing POI: {str(e)}")
        return pois


class SpatialProcessor:
    """Handles spatial operations"""

    @staticmethod
    def generate_candidates(grid: pd.Series) -> List[Dict]:
        try:
            roads = ox.graph_from_bbox(
                grid['max_lat'], grid['min_lat'],
                grid['max_lon'], grid['min_lon'],
                network_type='drive'
            )
            if not roads.edges:
                return []
            roads_proj = ox.project_graph(roads)
            nodes_proj = ox.graph_to_gdfs(roads_proj, nodes=True, edges=False)
            edges_proj = ox.graph_to_gdfs(roads_proj, nodes=False)
            candidates = []

            for _, edge in edges_proj.iterrows():
                if edge.geometry.length < 10:  # Skip short edges
                    continue
                for dist in np.linspace(0.2, 0.8, 4):  # Avoid edges near junctions
                    point_proj = edge.geometry.interpolate(dist, normalized=True)
                    start_node = nodes_proj.loc[edge.name[0]].geometry
                    end_node = nodes_proj.loc[edge.name[1]].geometry
                    if start_node.distance(point_proj) > CONFIG["junction_threshold_meters"] and \
                            end_node.distance(point_proj) > CONFIG["junction_threshold_meters"]:
                        point_wgs = ox.project_geometry(point_proj, crs=roads_proj.graph['crs'], to_crs='EPSG:4326')
                        candidates.append({
                            'lat': point_wgs.y,
                            'lon': point_wgs.x,
                            'grid_id': grid.name
                        })
            return candidates[:CONFIG["max_candidates_per_grid"]]
        except Exception as e:
            logger.error(f"Candidate generation error: {str(e)}")
            return []


class GNNTrainer:
    """Handles GNN training"""

    def __init__(self, in_channels: int, config: Dict):
        self.model = EnhancedGNN(in_channels, config["hidden_channels"], config["dropout"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.epochs = config["epochs"]

    def train(self, X: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor) -> EnhancedGNN:
        self.model.train()
        for _ in tqdm(range(self.epochs), desc="Training GNN"):
            self.optimizer.zero_grad()
            out = self.model(X, edge_index).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            self.optimizer.step()
        return self.model


def main():
    try:
        grids = DataLoader.load_grid_data("Training Data/city_grid_density.ods")
        existing_stops = gpd.read_file("Training Data/stib_stops.ods")
        existing_points = list(existing_stops[['stop_lat', 'stop_lon']].itertuples(index=False, name=None))
        poi_processor = POIProcessor(DataLoader.load_poi_config("poi_tags.json"))

        # Prepare features
        features = []
        logger.info("Processing grids...")
        for idx, grid in tqdm(grids.iterrows(), total=len(grids)):
            pois = poi_processor.fetch_pois(grid)
            grid_geometry = box(grid['min_lon'], grid['min_lat'], grid['max_lon'], grid['max_lat'])
            existing_count = sum(1 for pt in existing_points if grid_geometry.contains(Point(pt[1], pt[0])))

            features.append([
                grid['population_density'],
                len(pois),
                existing_count
            ])

        # Train model
        X = torch.tensor(StandardScaler().fit_transform(features), dtype=torch.float)
        y = torch.clamp(torch.tensor(grids['population_density'].values * 0.7 + np.random.rand(len(grids)) * 0.3), 0, 3)
        trainer = GNNTrainer(X.shape[1], CONFIG)
        model = trainer.train(X, torch.empty((2, 0), dtype=torch.long), y)

        # Generate candidates
        results = []
        spatial_processor = SpatialProcessor()
        for idx, grid in tqdm(grids.iterrows(), total=len(grids)):
            candidates = spatial_processor.generate_candidates(grid)
            for candidate in candidates:
                if not existing_points:
                    results.append(candidate)
                    continue
                min_distance = min(haversine((candidate['lat'], candidate['lon']), (lat, lon), unit='m')
                                   for lat, lon in existing_points)
                if min_distance > CONFIG["proximity_threshold_meters"]:
                    results.append(candidate)

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_excel(f"{CONFIG['output_dir']}/bus_stops.ods", engine='odf')
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")


if __name__ == "__main__":
    main()
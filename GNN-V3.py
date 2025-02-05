import json
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import requests
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from shapely.geometry import Point, box, Polygon
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pytz
import osmnx as ox
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import warnings
from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import Table, TableRow, TableCell
from odf.text import P

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
    "proximity_threshold": 0.00045,  # ~50 meters
    "max_candidates_per_grid": 5,
    "output_dir": "output",
    # Add default POI weights
    "default_poi_weights": {
        "amenity": {
            "hospital": 5,
            "school": 4,
            "university": 4,
            "bus_station": 5
        },
        "shop": {
            "supermarket": 3,
            "mall": 4
        },
        "public_transport": {
            "station": 4
        }
    }
}

# Create output directory
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
        """Load and validate grid data"""
        try:
            df = pd.read_excel(file_path, engine='odf')
            df = df.rename(columns={'density_rank': 'population_density'})

            # Validate columns
            required_columns = ['min_lat', 'min_lon', 'max_lat', 'max_lon', 'population_density']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")

            # Coordinate validation and normalization
            for lat_col in ['min_lat', 'max_lat']:
                df[lat_col] = df[lat_col].clip(-90, 90)
            for lon_col in ['min_lon', 'max_lon']:
                df[lon_col] = df[lon_col].clip(-180, 180)

            # Ensure proper coordinate order
            df['min_lat'], df['max_lat'] = np.where(
                df['min_lat'] < df['max_lat'],
                (df['min_lat'], df['max_lat']),
                (df['max_lat'], df['min_lat'])
            )
            df['min_lon'], df['max_lon'] = np.where(
                df['min_lon'] < df['max_lon'],
                (df['min_lon'], df['max_lon']),
                (df['max_lon'], df['min_lon'])
            )

            # Data validation
            valid = (
                    (df['min_lat'] < df['max_lat']) &
                    (df['min_lon'] < df['max_lon']) &
                    (df['population_density'].between(1, 5))
            )
            df = df[valid].reset_index(drop=True)

            if df.empty:
                raise ValueError("No valid grids after processing")

            logger.info(f"Loaded {len(df)} valid grids")
            return df

        except Exception as e:
            logger.error(f"Error loading grid data: {str(e)}")
            raise

    @staticmethod
    def load_poi_config(file_path: str) -> Dict:
        """Load POI configuration with proper error handling"""
        try:
            with open(file_path, "r") as f:
                poi_config = json.load(f)

            if not isinstance(poi_config, dict):
                raise ValueError("POI config should be a dictionary")

            # Validate structure
            if not all(isinstance(v, dict) for v in poi_config.values()):
                raise ValueError("Invalid POI config structure")

            return poi_config
        except Exception as e:
            logger.error(f"Using default POI config: {str(e)}")
            return CONFIG["default_poi_weights"]


class POIProcessor:
    """Handles POI data processing and counting"""

    def __init__(self, poi_config: Dict):
        self.poi_config = poi_config
        self.all_poi_types = self._get_all_poi_types()

    def _get_all_poi_types(self) -> List[str]:
        """Get list of all POI types from config"""
        return list({
            poi_type
            for category in self.poi_config.values()
            for poi_type in category.keys()
        })

    def count_pois(self, pois: List[Dict]) -> Dict[str, int]:
        """Count occurrences of each POI type in a grid"""
        counts = {poi_type: 0 for poi_type in self.all_poi_types}

        for poi in pois:
            for category, tags in self.poi_config.items():
                if category in poi['tags']:
                    tag_value = poi['tags'][category]
                    if tag_value in tags:
                        counts[tag_value] += 1
        return counts

    def calculate_poi_score(self, pois: List[Dict]) -> int:
        """Calculate weighted POI score using OSM tags"""
        if not isinstance(pois, list):
            return 0

        score = 0
        for poi in pois:
            for category, tags in self.poi_config.items():
                # Check if POI has this category tag and it matches our configuration
                if category in poi['tags']:
                    tag_value = poi['tags'][category]
                    if tag_value in tags:
                        score += tags[tag_value]
        return score

    def fetch_pois(self, grid: Dict) -> List[Dict]:
        """Fetch POIs for a grid using OSMnx-compatible syntax"""
        try:
            # Define bounding box as (north, south, east, west)
            bbox = (
                grid['max_lat'],  # North
                grid['min_lat'],  # South
                grid['max_lon'],  # East
                grid['min_lon']  # West
            )

            # Convert POI config to OSMnx-compatible format
            osm_tags = {
                category: list(poi_types.keys())
                for category, poi_types in self.poi_config.items()
            }

            # Fetch POIs with corrected parameter order
            pois_gdf = ox.features.features_from_bbox(
                *bbox,  # Unpack bbox tuple
                tags=osm_tags
            )
            return self._process_pois(pois_gdf, grid)
        except Exception as e:
            logger.error(f"POI fetch error: {str(e)}")
            return []

    def _process_pois(self, pois_gdf: gpd.GeoDataFrame, grid: Dict) -> List[Dict]:
        """Process raw POI data"""
        pois = []
        for _, row in pois_gdf.iterrows():
            try:
                geom = row.geometry
                if geom.geom_type not in ['Point', 'Polygon']:
                    continue

                poi_data = {
                    'lat': geom.centroid.y if geom.geom_type == 'Polygon' else geom.y,
                    'lon': geom.centroid.x if geom.geom_type == 'Polygon' else geom.x,
                    'tags': {k: v for k, v in row.items() if k in self.poi_config}
                }
                pois.append(poi_data)
            except Exception as e:
                logger.warning(f"Error processing POI: {str(e)}")
        return pois


class GNNTrainer:
    """Handles GNN training"""

    def __init__(self, in_channels: int, config: Dict):
        self.model = EnhancedGNN(
            in_channels=in_channels,
            hidden_channels=config["hidden_channels"],
            dropout=config["dropout"]
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=1e-4
        )
        self.epochs = config["epochs"]

    def train(self, X: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor) -> EnhancedGNN:
        """Train the GNN model"""
        self.model.train()
        pbar = tqdm(range(self.epochs), desc="Training GNN")

        for epoch in pbar:
            self.optimizer.zero_grad()
            out = self.model(X, edge_index).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            self.optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})
        return self.model


class SpatialProcessor:
    """Handles spatial operations"""

    @staticmethod
    def create_spatial_graph(grids: pd.DataFrame) -> torch.Tensor:
        """Create graph edges based on grid centroids"""
        centroids = grids.apply(lambda row: [
            (row['min_lat'] + row['max_lat']) / 2,
            (row['min_lon'] + row['max_lon']) / 2
        ], axis=1).tolist()

        edges = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                if np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j])) < 0.02:
                    edges.extend([[i, j], [j, i]])

        return torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0),
                                                                                                dtype=torch.long)

    @staticmethod
    def generate_candidates(grid: pd.Series) -> List[Dict]:
        """Generate candidate bus stop locations"""
        try:
            roads = ox.graph_from_bbox(
                grid['max_lat'], grid['min_lat'],
                grid['max_lon'], grid['min_lon'],
                network_type='drive'
            )
            edges = ox.graph_to_gdfs(roads, nodes=False)
            candidates = []

            for _, edge in edges.iterrows():
                if 'geometry' in edge and edge.geometry.length > 0.0001:
                    for dist in np.linspace(0.1, 0.9, 5):
                        point = edge.geometry.interpolate(dist, normalized=True)
                        candidates.append({
                            'lat': point.y,
                            'lon': point.x,
                            'grid_id': grid.name
                        })
            return candidates[:CONFIG["max_candidates_per_grid"]]
        except Exception as e:
            logger.error(f"Candidate generation error: {str(e)}")
            return []


class OutputGenerator:
    """Handles output generation"""

    @staticmethod
    def save_to_ods(data: pd.DataFrame, filename: str):
        """Save data to ODS file"""
        try:
            doc = OpenDocumentSpreadsheet()
            table = Table(name="BusStops")

            # Create header
            tr = TableRow()
            for col in data.columns:
                tc = TableCell()
                tc.addElement(P(text=col))
                tr.addElement(tc)
            table.addElement(tr)

            # Add data rows
            for _, row in data.iterrows():
                tr = TableRow()
                for val in row:
                    tc = TableCell()
                    tc.addElement(P(text=str(val)))
                    tr.addElement(tc)
                table.addElement(tr)

            doc.spreadsheet.addElement(table)
            doc.save(f"{CONFIG['output_dir']}/{filename}")
            logger.info(f"Saved ODS file: {filename}")
        except Exception as e:
            logger.error(f"Error saving ODS file: {str(e)}")

    @staticmethod
    def create_map(data: pd.DataFrame, grids: pd.DataFrame, filename: str):
        """Create interactive Folium map"""
        try:
            m = folium.Map(
                location=[data['lat'].mean(), data['lon'].mean()],
                zoom_start=13,
                tiles='CartoDB positron'
            )

            # Add bus stops
            for _, row in data.iterrows():
                folium.Marker(
                    [row['lat'], row['lon']],
                    popup=f"Grid {row['grid_id']} - Score: {row['score']}",
                    icon=folium.Icon(color='green', icon='bus', prefix='fa')
                ).add_to(m)

            # Add grid boundaries
            for _, grid in grids.iterrows():
                folium.Rectangle(
                    bounds=[
                        [grid['min_lat'], grid['min_lon']],
                        [grid['max_lat'], grid['max_lon']]
                    ],
                    color='#ff7800',
                    fill=True,
                    fill_color='#ffff00',
                    fill_opacity=0.2
                ).add_to(m)

            m.save(f"{CONFIG['output_dir']}/{filename}")
            logger.info(f"Saved map: {filename}")
        except Exception as e:
            logger.error(f"Error creating map: {str(e)}")


def main():
    """Main execution flow"""
    try:
        # Initialize components
        grids = DataLoader.load_grid_data("Training Data/city_grid_density.ods")
        existing_stops = pd.read_excel("Training Data/stib_stops.ods", engine='odf')
        existing_stops = gpd.GeoDataFrame(
            existing_stops,
            geometry=gpd.points_from_xy(
                existing_stops['stop_lon'],
                existing_stops['stop_lat']
            ),
            crs="EPSG:4326"
        )
        poi_processor = POIProcessor(DataLoader.load_poi_config("poi_tags.json"))
        spatial_processor = SpatialProcessor()
        output_generator = OutputGenerator()

        # Get environmental context
        raining, temp = get_weather("Brussels")
        is_day, is_peak = process_time("17:30")

        # Prepare features and counts
        features = []
        poi_counts_list = []

        logger.info("Processing grids...")
        for idx, grid in tqdm(grids.iterrows(), total=len(grids)):
            grid_geometry = box(
                grid['min_lon'], grid['min_lat'],
                grid['max_lon'], grid['max_lat']
            )

            pois = poi_processor.fetch_pois(grid)
            poi_counts = poi_processor.count_pois(pois)
            poi_counts_list.append(poi_counts)

            existing_count = existing_stops[existing_stops.geometry.within(grid_geometry)].shape[0]

            features.append([
                grid['population_density'],
                sum(poi_counts.values()),  # Total POI count as feature
                int(is_day),
                int(is_peak),
                int(raining),
                temp,
                existing_count
            ])

        # Prepare data for model
        X = torch.tensor(StandardScaler().fit_transform(features), dtype=torch.float)
        edge_index = spatial_processor.create_spatial_graph(grids)
        y = synthetic_targets(grids)

        # Train model
        trainer = GNNTrainer(X.shape[1], CONFIG)
        model = trainer.train(X, edge_index, y)

        # Generate predictions
        predictions = predict(model, X, edge_index)

        # Generate candidate points
        results = []
        logger.info("Generating candidates...")
        for idx, grid in tqdm(grids.iterrows(), total=len(grids)):
            candidates = spatial_processor.generate_candidates(grid)
            selected = []
            current_counts = poi_counts_list[idx]

            for candidate in candidates:
                candidate_pt = Point(candidate['lon'], candidate['lat'])
                distances = existing_stops.geometry.distance(candidate_pt)

                if distances.min() > CONFIG["proximity_threshold"]:
                    selected.append({
                        **candidate,
                        'score': predictions[idx],
                        **current_counts  # Add POI counts to candidate data
                    })

            results.extend(selected[:predictions[idx]])

        # Save results
        results_df = pd.DataFrame(results)
        output_generator.save_to_ods(results_df, "bus_stops.ods")
        output_generator.create_map(results_df, grids, "bus_stops_map.html")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise


def synthetic_targets(grids: pd.DataFrame) -> torch.Tensor:
    """Generate synthetic training targets"""
    base = grids['population_density'].values * 0.7
    noise = np.random.rand(len(grids)) * 0.3
    return torch.tensor(np.clip(base + noise, 0, 3), dtype=torch.float)


def predict(model: EnhancedGNN, X: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    """Generate predictions from trained model"""
    model.eval()
    with torch.no_grad():
        raw_pred = model(X, edge_index).squeeze().numpy()
    return np.clip(raw_pred * 3, 0, 3).astype(int)


def get_weather(city: str) -> Tuple[bool, float]:
    # Load the API keys
    with open('api_keys.json') as json_file:
        api_keys = json.load(json_file)

    # Google Maps API key
    WEATHER_API_KEY = api_keys['Weather_API']['API_key']
    """Get current weather conditions"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}"
        response = requests.get(url).json()
        temp = response['main']['temp'] - 273.15
        raining = any('rain' in w['main'].lower() for w in response.get('weather', []))
        return raining, temp
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return False, 20.0


def process_time(time_str: str) -> Tuple[bool, bool]:
    """Determine time-based features"""
    try:
        tz = pytz.timezone('Europe/Brussels')
        current_time = datetime.now(tz).time()
        is_day = 7 <= current_time.hour < 19
        is_peak = (8 <= current_time.hour < 10) or (17 <= current_time.hour < 19)
        return is_day, is_peak
    except Exception as e:
        logger.error(f"Time processing error: {str(e)}")
        return True, False


if __name__ == "__main__":
    main()
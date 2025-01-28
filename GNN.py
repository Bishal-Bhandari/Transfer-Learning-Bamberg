import pandas as pd
import osmnx as ox
import folium
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from scipy.spatial import cKDTree
from tqdm import tqdm
import geopandas as gpd
from joblib import Parallel, delayed
from shapely.geometry import Polygon, MultiPolygon

ox.settings.log_console = True
ox.settings.use_cache = True  # Enable caching for OSM data
tqdm.pandas()


# 1. Optimized Data Loading
def load_data(file_path):
    """Load data with optimized dtype handling"""
    df = pd.read_excel(file_path, engine='odf',
                       dtype={'Latitude': np.float32, 'Longitude': np.float32})
    scaler = MinMaxScaler()
    df['Normalized_Density'] = scaler.fit_transform(df[['Density']]).astype(np.float32)
    return df


# 2. Enhanced OSM Data Integration with Caching
def get_osm_data(place_name="Bamberg, Germany"):
    """Retrieve OSM data with efficient geometry handling"""
    # Get road network with simplified graph
    graph = ox.graph_from_place(place_name, network_type='drive', simplify=True)
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)

    # Get POIs using predefined categories
    poi_tags = {
        'amenity': ['school', 'hospital', 'restaurant', 'cafe'],
        'shop': True,
        'public_transport': ['station'],
        'tourism': ['hotel', 'attraction']
    }

    pois = ox.features_from_place(place_name, poi_tags)
    pois = pois[pois.geometry.notnull()].copy()

    # Updated geometry handling with proper type checking
    def convert_geometry(g):
        if isinstance(g, (Polygon, MultiPolygon)):
            return g.centroid
        return g

    pois['geometry'] = pois.geometry.apply(convert_geometry)
    pois = pois.explode(index_parts=True).reset_index(drop=True)
    pois = pois[pois.geometry.type == 'Point'].copy()

    return graph, nodes, pois


# 3. Parallel POI Density Calculation
def calculate_poi_density_parallel(candidates, pois, radius=500, n_jobs=-1):
    """Parallel POI density calculation with UTM projection"""
    if pois.empty:
        candidates['POI_Count'] = 0
        return candidates

    # Convert to GeoDataFrames
    candidates_gdf = gpd.GeoDataFrame(
        candidates,
        geometry=gpd.points_from_xy(candidates.Longitude, candidates.Latitude),
        crs="EPSG:4326"
    )

    # Find UTM CRS using GeoPandas
    utm_crs = candidates_gdf.estimate_utm_crs()

    # Project to UTM
    candidates_utm = candidates_gdf.to_crs(utm_crs)
    pois_utm = pois.to_crs(utm_crs)

    # Create spatial index
    poi_coords = np.array([(geom.x, geom.y) for geom in pois_utm.geometry])
    tree = cKDTree(poi_coords)

    def count_pois(row):
        point = row.geometry
        return len(tree.query_ball_point([point.x, point.y], r=radius))

    # Parallel processing
    candidates_utm['POI_Count'] = Parallel(n_jobs=n_jobs)(
        delayed(count_pois)(row) for _, row in tqdm(candidates_utm.iterrows(), total=len(candidates_utm))
    )

    # Merge results back
    return candidates.merge(
        candidates_utm[['POI_Count']],
        left_index=True,
        right_index=True
    )


# 4. Optimized Candidate Generation using Road Network
def generate_candidates(graph, existing_stops, search_radius_km=5):
    """Generate candidates based on road network nodes with proper node ID handling"""
    # Convert existing stops to nodes
    if 'node_id' not in existing_stops.columns:
        # Add node_id to existing stops if missing
        existing_stops['geometry'] = gpd.points_from_xy(
            existing_stops.Longitude,
            existing_stops.Latitude
        )
        existing_stops['node_id'] = ox.distance.nearest_nodes(
            graph,
            existing_stops.Longitude,
            existing_stops.Latitude
        )

    # Get all graph nodes
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)

    # Filter out nodes used by existing stops
    candidate_nodes = nodes[~nodes.index.isin(existing_stops['node_id'])]

    # Create candidates from remaining nodes
    candidates = candidate_nodes.sample(frac=0.3).reset_index()[['y', 'x']]
    candidates.columns = ['Latitude', 'Longitude']

    return candidates


# 5. Vectorized Graph Data Preparation
def prepare_graph_data(graph, combined):
    """Add node degree as additional feature"""
    # Calculate node degrees
    degrees = pd.Series(dict(graph.degree())).rename('degree')

    # Create feature matrix
    node_features = combined.groupby('node_id').agg({
        'Normalized_Density': 'mean',
        'POI_Count': 'mean'
    })

    # Add degree feature
    node_features = node_features.join(degrees).fillna(0)

    # Normalize features
    scaler = MinMaxScaler()
    node_features = scaler.fit_transform(node_features)

    # Create edge index
    edge_index = torch.tensor(
        [(list(graph.nodes()).index(u), list(graph.nodes()).index(v))
         for u, v in graph.edges()],
        dtype=torch.long
    ).t().contiguous()

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index
    ), combined


# 6. Enhanced GNN Model with Regularization
class BusStopPredictor(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim//2)
        self.dropout = torch.nn.Dropout(0.3)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim//2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv3(x, edge_index))
        return torch.sigmoid(self.predictor(x))


# 7. Optimized Training Loop with Early Stopping
def train_model(data, epochs=500, patience=20):
    model = BusStopPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    target_weights = data.x[:, 0] * 0.6 + data.x[:, 1] * 0.4  # Adjusted weights

    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(data).squeeze()
        loss = torch.mean((pred - target_weights) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss < best_loss:
            best_loss = loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} with loss {loss:.4f}")
            break

    return model


# 8. Optimized Prediction Handling
def predict_and_adjust(model, data, graph, candidates):
    """Proper prediction alignment with candidate nodes"""
    model.eval()
    with torch.no_grad():
        predictions = model(data).squeeze().numpy()

    # Get the full node list from graph data preparation
    all_nodes = list(graph.nodes())

    # Map predictions to candidates using node_id
    candidates['pred_prob'] = candidates['node_id'].apply(
        lambda nid: predictions[all_nodes.index(nid)] if nid in all_nodes else 0.0
    )

    # Filter and adjust
    final_stops = candidates[candidates['pred_prob'] > 0.5]

    if final_stops.empty:
        final_stops = candidates.nlargest(1, 'pred_prob')

    return final_stops


# Main Workflow
def main():
    # Load data
    bus_stops = load_data("Training Data/final_busStop_density.ods")

    # Get OSM data
    graph, _, pois = get_osm_data()

    # Generate candidates with proper node ID handling
    candidates = generate_candidates(graph, bus_stops)
    candidates = calculate_poi_density_parallel(candidates, pois)

    # Prepare graph data (now includes existing stops)
    combined = pd.concat([bus_stops, candidates], ignore_index=True)
    graph_data, combined = prepare_graph_data(graph, combined)

    # Train model
    model = train_model(graph_data)

    # Predict and visualize
    predictions = predict_and_adjust(model, graph_data, graph, combined)

    # Create optimized visualization
    m = folium.Map(location=[49.8988, 10.9028], zoom_start=14)
    for _, stop in predictions.iterrows():
        folium.CircleMarker(
            location=[stop.Latitude, stop.Longitude],
            radius=6,
            color='#ff0000' if stop.pred_prob > 0.7 else '#ffa500',
            fill=True,
            opacity=0.7,
            popup=f"Score: {stop.pred_prob:.2f}"
        ).add_to(m)
    m.save("optimized_predictions.html")


if __name__ == "__main__":
    main()
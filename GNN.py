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

tqdm.pandas()


# 1. Data Loading with Enhanced Features
def load_data(file_path):
    """Load bus stop data with density information"""
    df = pd.read_excel(file_path, engine='odf')
    scaler = MinMaxScaler()
    df['Normalized_Density'] = scaler.fit_transform(df[['Density']])
    return df


# 2. OSM Data Integration with POI Collection
def get_osm_data(place_name="Bamberg, Germany"):
    """Retrieve road network and POI data from OSM"""
    # Get road network
    graph = ox.graph_from_place(place_name, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(graph)

    # Get POIs - expanded categories
    poi_tags = {
        'amenity': ['school', 'hospital', 'restaurant', 'cafe', 'library'],
        'shop': ['supermarket', 'convenience'],
        'public_transport': ['station', 'stop_position'],
        'office': ['company', 'government'],
        'tourism': ['hotel', 'attraction'],
        'leisure': ['park', 'sports_centre']
    }
    pois = ox.features_from_place(place_name, poi_tags)
    return graph, nodes, pois


# 3. Feature Engineering with POI Density
def calculate_poi_density(candidates, pois, radius=500):
    """Calculate POI density around candidate points with proper geometry handling"""
    # Convert radius to approximate degrees
    radius_deg = radius / 111139

    # Extract coordinates from all geometry types
    poi_coords = []
    for geom in pois.geometry:
        if geom.geom_type == 'Point':
            poi_coords.append((geom.y, geom.x))
        elif geom.geom_type in ['MultiPoint', 'LineString', 'MultiLineString']:
            for point in geom.geoms:
                if hasattr(point, 'coords'):
                    poi_coords.append((point.y, point.x))
        elif geom.geom_type in ['Polygon', 'MultiPolygon']:
            # Use centroid for area features
            centroid = geom.centroid
            poi_coords.append((centroid.y, centroid.x))

    if not poi_coords:  # Handle case with no POIs
        candidates['POI_Count'] = 0
        return candidates

    # Create spatial index
    tree = cKDTree(poi_coords)

    # Query POIs for each candidate
    counts = tree.query_ball_point(
        candidates[['Latitude', 'Longitude']].values,
        r=radius_deg,
        return_length=True
    )

    candidates['POI_Count'] = counts
    return candidates


# 4. Enhanced Candidate Generation
def generate_candidates(df, place_center, search_radius_km=5):
    """Generate potential bus stop locations considering existing density"""
    # Create grid covering the search area
    lat_range = search_radius_km / 111
    lon_range = search_radius_km / (111 * np.cos(np.radians(place_center[0])))

    return pd.DataFrame([
        (lat, lon)
        for lat in np.linspace(place_center[0] - lat_range, place_center[0] + lat_range, 50)
        for lon in np.linspace(place_center[1] - lon_range, place_center[1] + lon_range, 50)
    ], columns=['Latitude', 'Longitude'])


# 5. Graph Data Preparation with Dual Features
def prepare_graph_data(graph, nodes, candidates):
    """Create graph data structure with proper node ID handling"""
    # Step 1: Add node_id column with error handling
    try:
        # Verify coordinate columns exist
        if not {'Latitude', 'Longitude'}.issubset(candidates.columns):
            raise KeyError("Missing Latitude/Longitude columns in candidates DataFrame")

        # Snap candidates to road network with progress indication
        print("Snapping candidates to road network...")
        candidates['node_id'] = candidates.progress_apply(
            lambda r: ox.distance.nearest_nodes(
                graph,
                X=r['Longitude'],  # Explicit column access
                Y=r['Latitude']
            ),
            axis=1
        )
    except Exception as e:
        print(f"Error creating node_id: {e}")
        raise

    # Step 2: Handle nodes not found in the graph
    print(f"Original candidates: {len(candidates)}")
    candidates = candidates.dropna(subset=['node_id']).copy()
    print(f"Candidates after node matching: {len(candidates)}")

    # Step 3: Create unified node index
    all_node_ids = set(graph.nodes()).union(set(candidates['node_id']))
    node_id_map = {orig: idx for idx, orig in enumerate(sorted(all_node_ids))}

    # Step 4: Create filtered edge list
    edge_list = [
        (node_id_map[u], node_id_map[v])
        for u, v in graph.edges()
        if u in node_id_map and v in node_id_map
    ]

    # Step 5: Create node features matrix
    node_features = pd.DataFrame(index=sorted(all_node_ids))
    node_features['density'] = 0.0
    node_features['poi'] = 0.0

    # Aggregate candidate features
    agg_features = candidates.groupby('node_id')[['Normalized_Density', 'POI_Count']].mean()
    node_features.update(agg_features)

    # Normalize features
    node_features.fillna(0, inplace=True)
    node_features = MinMaxScaler().fit_transform(node_features)

    # Convert to tensors
    return (
        Data(x=torch.tensor(node_features, dtype=torch.float32),
             edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous()),
        candidates,
        node_id_map
    )


# 6. Enhanced GNN Model
class BusStopPredictor(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.predictor = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.predictor(x))


# 7. Model Training with Combined Features
def train_model(data, epochs=200):
    model = BusStopPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create synthetic targets based on feature combination
    target_weights = data.x[:, 0] * 0.7 + data.x[:, 1] * 0.3  # Density 70%, POI 30%

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(data).squeeze()
        loss = torch.mean((pred - target_weights) ** 2)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    return model


# 8. Prediction and Post-Processing
def predict_and_adjust(model, data, graph, candidates, node_id_map):
    """Generate predictions with proper index mapping"""
    model.eval()
    with torch.no_grad():
        predictions = model(data).squeeze().numpy()

    # Map predictions using node_id_map
    candidates['pred_prob'] = candidates['node_id'].apply(
        lambda x: predictions[node_id_map[x]] if x in node_id_map else 0.0
    )

    # Filter candidates based on predicted probability
    final_stops = candidates[candidates['pred_prob'] > 0.5].copy()

    # Check if final_stops is empty
    if final_stops.empty:
        print("No candidates met the probability threshold. Adjusting to use the top candidate...")
        # Select the top candidate based on prediction probability
        top_candidate_idx = candidates['pred_prob'].idxmax()
        final_stops = candidates.loc[[top_candidate_idx]].copy()

    # Snap to nearest road nodes (vectorized approach)
    final_stops['adjusted_node'] = ox.distance.nearest_nodes(
        graph, X=final_stops['Longitude'].values, Y=final_stops['Latitude'].values
    )

    return final_stops


# Main Workflow
def main():
    # Load and prepare data
    bus_stops = load_data("Training Data/final_busStop_density.ods")
    graph, nodes, pois = get_osm_data()

    # Generate and enhance candidates
    candidates = generate_candidates(bus_stops, (49.8988, 10.9028))
    candidates = calculate_poi_density(candidates, pois)

    # Merge existing bus stops with candidates
    combined = pd.concat([bus_stops, candidates], ignore_index=True)

    # Prepare graph data
    graph_data, combined, node_id_map = prepare_graph_data(graph, nodes, combined)

    # Train and predict
    model = train_model(graph_data)
    predictions = predict_and_adjust(model, graph_data, graph, combined, node_id_map)

    # Visualize results
    m = folium.Map(location=[49.8988, 10.9028], zoom_start=14)
    for _, stop in predictions.iterrows():
        folium.CircleMarker(
            location=[stop.Latitude, stop.Longitude],
            radius=5,
            color='green' if stop.pred_prob > 0.7 else 'orange',
            fill=True,
            popup=f"Prob: {stop.pred_prob:.2f}<br>POIs: {stop.POI_Count}"
        ).add_to(m)
    m.save("bus_stop_predictions.html")


if __name__ == "__main__":
    main()

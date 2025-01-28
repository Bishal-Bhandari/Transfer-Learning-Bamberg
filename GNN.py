import pandas as pd
import osmnx as ox
import folium
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from scipy.spatial import cKDTree


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
    """Calculate POI density around candidate points"""
    # Convert radius to approximate degrees
    radius_deg = radius / 111139

    # Get POI coordinates
    poi_coords = np.array([[point.y, point.x] for geom in pois.geometry
                           for point in (geom if geom.type == 'MultiPoint' else [geom])])

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
    """Create graph data structure with density and POI features"""
    # Snap candidates to road network
    candidates['node_id'] = candidates.apply(
        lambda r: ox.distance.nearest_nodes(graph, r.Longitude, r.Latitude),
        axis=1
    )

    # Aggregate features per road node
    node_features = nodes[['x', 'y']].copy()
    node_features['density'] = 0.0
    node_features['poi'] = 0.0

    # Calculate mean features for nodes with multiple candidates
    agg_features = candidates.groupby('node_id')[['Normalized_Density', 'POI_Count']].mean()

    for node_id, features in agg_features.iterrows():
        if node_id in node_features.index:
            node_features.at[node_id, 'density'] = features['Normalized_Density']
            node_features.at[node_id, 'poi'] = features['POI_Count']

    # Normalize features
    scaler = MinMaxScaler()
    node_features[['density', 'poi']] = scaler.fit_transform(node_features[['density', 'poi']])

    # Create edge index
    edge_index = torch.tensor(
        [(u, v) for u, v in graph.edges() if u in node_features.index and v in node_features.index],
        dtype=torch.long
    ).t().contiguous()

    return Data(
        x=torch.tensor(node_features[['density', 'poi']].values, dtype=torch.float),
        edge_index=edge_index
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
def predict_and_adjust(model, data, graph, candidates):
    """Generate predictions and snap to road network"""
    model.eval()
    with torch.no_grad():
        predictions = model(data).squeeze().numpy()

    # Assign predictions to candidates
    candidates['pred_prob'] = predictions[candidates['node_id'].apply(lambda x: list(data.x[:, 0]).index(x))]

    # Filter and adjust to road network
    final_stops = candidates[candidates['pred_prob'] > 0.5]
    final_stops['adjusted_coords'] = final_stops.apply(
        lambda r: ox.distance.nearest_nodes(graph, r.Longitude, r.Latitude),
        axis=1
    )

    return final_stops


# Main Workflow
def main():
    # Load and prepare data
    bus_stops = load_data("Training Data/final_busStop_density.ods")
    graph, pois = get_osm_data()

    # Generate and enhance candidates
    candidates = generate_candidates(bus_stops, (49.8988, 10.9028))
    candidates = calculate_poi_density(candidates, pois)

    # Merge existing bus stops with candidates
    combined = pd.concat([bus_stops, candidates], ignore_index=True)

    # Prepare graph data
    graph_data = prepare_graph_data(graph, combined)

    # Train and predict
    model = train_model(graph_data)
    predictions = predict_and_adjust(model, graph_data, graph, candidates)

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

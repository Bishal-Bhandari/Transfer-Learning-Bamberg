import pandas as pd
import osmnx as ox
import folium
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from scipy.spatial import cKDTree


# Load and Normalize Data
def load_data(file_path):
    df = pd.read_excel(file_path, engine='odf')
    scaler = MinMaxScaler()
    df['Normalized_Density'] = scaler.fit_transform(df[['Density']])
    return df


# Load OSM Data and POIs
def get_osm_data():
    place_name = "Bamberg, Germany"
    graph = ox.graph_from_place(place_name, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(graph)

    # Get POI data
    tags = {
        'amenity': True,
        'public_transport': ['station', 'stop_position'],
        'shop': ['supermarket', 'mall'],
        'office': True,
        'tourism': True
    }
    pois = ox.features_from_place(place_name, tags)
    return graph, nodes, edges, pois


# Generate Candidate Locations
def generate_candidate_locations(df, center_lat, center_lon, radius_km, high_density_threshold=0.7, base_points=10):
    lat_radius = radius_km / 111
    lon_radius = radius_km / (111 * np.cos(np.radians(center_lat)))

    high_density = df[df['Normalized_Density'] >= high_density_threshold]
    low_density = df[df['Normalized_Density'] < high_density_threshold]

    high_density_candidates = pd.DataFrame([
        (lat, lon) for lat in np.linspace(center_lat - lat_radius, center_lat + lat_radius, base_points * 2)
        for lon in np.linspace(center_lon - lon_radius, center_lon + lon_radius, base_points * 2)
    ], columns=["Latitude", "Longitude"])

    low_density_candidates = low_density[['Latitude', 'Longitude']].drop_duplicates()
    all_candidates = pd.concat([high_density_candidates, low_density_candidates], ignore_index=True)
    return all_candidates


# Assign Density and POI counts to Candidates
def assign_features_to_candidates(candidates, df, pois):
    # Assign density
    density_tree = cKDTree(df[['Latitude', 'Longitude']])
    distances, indices = density_tree.query(candidates[['Latitude', 'Longitude']], k=1)
    candidates['Density'] = df.iloc[indices]['Normalized_Density'].values

    # Assign POI counts
    def get_poi_coords(pois_gdf):
        return np.array([(geom.y, geom.x) for geom in pois_gdf.geometry if geom.geom_type == 'Point'])

    poi_coords = get_poi_coords(pois)
    if len(poi_coords) > 0:
        poi_tree = cKDTree(poi_coords)
        radius_deg = 500 / 111000  # ~500 meters
        poi_counts = poi_tree.query_ball_point(candidates[['Latitude', 'Longitude']].values, r=radius_deg,
                                               return_length=True)
        candidates['POI_Count'] = poi_counts
    else:
        candidates['POI_Count'] = 0

    # Normalize POI counts
    scaler = MinMaxScaler()
    candidates['POI_Normalized'] = scaler.fit_transform(candidates[['POI_Count']])
    return candidates


# Prepare Graph Data with Candidates
def prepare_graph_data_with_candidates(df, graph, nodes, candidates):
    combined_df = pd.concat([df, candidates], ignore_index=True)
    combined_df['Node_ID'] = combined_df.apply(
        lambda row: ox.distance.nearest_nodes(graph, row['Longitude'], row['Latitude']), axis=1
    )
    combined_df['Node_ID'] = combined_df['Node_ID'].astype(int)

    node_features = pd.DataFrame(index=nodes.index)
    node_features['Density'] = 0.0
    node_features['POI_Normalized'] = 0.0

    for _, row in combined_df.iterrows():
        node_id = row['Node_ID']
        if node_id in node_features.index:
            node_features.at[node_id, 'Density'] = row['Density']
            node_features.at[node_id, 'POI_Normalized'] = row['POI_Normalized']

    valid_indices = list(node_features.index)
    filtered_edges = [(u, v) for u, v, *_ in graph.edges if u in valid_indices and v in valid_indices]
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(valid_indices)}
    reindexed_edges = [(node_id_map[u], node_id_map[v]) for u, v in filtered_edges]

    edge_index = torch.tensor(reindexed_edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features[['Density', 'POI_Normalized']].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data, combined_df


# Define GNN Model
class GCN(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=1, num_layers=3):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)


# Train GNN
def train_gnn(data, df, epochs=500, lr=0.01):
    model = GCN(input_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create composite target from features
    composite_target = data.x[:, 0] + data.x[:, 1]  # Density + POI

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data).squeeze()

        # Weight by composite importance
        weights = composite_target + 0.1  # Ensure low-weight areas contribute
        loss = (weights * (out - composite_target) ** 2).mean()

        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    return model


# Predict Candidates
def predict_candidates(model, data, candidates):
    model.eval()
    with torch.no_grad():
        predictions = model(data).squeeze().numpy()
        candidates['Prediction'] = predictions[-len(candidates):]
        candidates['Predicted_Stop'] = (candidates['Prediction'] > 0.5).astype(int)
    return candidates


# Adjust predictions to nearest road nodes
def adjust_predictions_to_road(candidates, graph):
    candidates['Adjusted_Node_ID'] = candidates.apply(
        lambda row: ox.distance.nearest_nodes(graph, row['Longitude'], row['Latitude']), axis=1
    )
    candidates['Adjusted_Latitude'] = candidates['Adjusted_Node_ID'].map(
        lambda node_id: graph.nodes[node_id]['y'] if node_id in graph.nodes else np.nan
    )
    candidates['Adjusted_Longitude'] = candidates['Adjusted_Node_ID'].map(
        lambda node_id: graph.nodes[node_id]['x'] if node_id in graph.nodes else np.nan
    )
    return candidates.dropna(subset=['Adjusted_Latitude', 'Adjusted_Longitude'])


# Visualize Results
def visualize_candidate_predictions(candidates, output_html):
    map_bamberg = folium.Map(location=[49.8988, 10.9028], zoom_start=13)
    for _, row in candidates.iterrows():
        color = 'green' if row['Predicted_Stop'] else 'gray'
        folium.CircleMarker(
            location=[row['Adjusted_Latitude'], row['Adjusted_Longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Density: {row['Density']:.2f}, POI: {row['POI_Count']}"
        ).add_to(map_bamberg)
    map_bamberg.save(output_html)


# Main Function
def main():
    input_file = "Training Data/final_busStop_density.ods"
    output_file = "Model Data/GNN-predicted_candidates.ods"
    output_html = "Template/GNN-candidate_predictions.html"

    df = load_data(input_file)
    graph, nodes, edges, pois = get_osm_data()
    candidates = generate_candidate_locations(df, 49.8988, 10.9028, 5)
    candidates = assign_features_to_candidates(candidates, df, pois)
    data, combined_df = prepare_graph_data_with_candidates(df, graph, nodes, candidates)
    model = train_gnn(data, df)
    candidates = predict_candidates(model, data, candidates)
    candidates = adjust_predictions_to_road(candidates, graph)
    candidates.to_excel(output_file, engine='odf')
    visualize_candidate_predictions(candidates, output_html)
    print("Processing complete")


if __name__ == "__main__":
    main()

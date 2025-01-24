import pandas as pd
import osmnx as ox
import folium
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.spatial import cKDTree


# Load and Normalize Data
def load_data(file_path):
    df = pd.read_excel(file_path, engine='odf')
    scaler = MinMaxScaler()
    df['Normalized_Density'] = scaler.fit_transform(df[['Density']])
    return df


# Load OSM Data
def get_osm_data():
    graph = ox.graph_from_place("Bamberg, Germany", network_type='drive')
    nodes, edges = ox.graph_to_gdfs(graph)
    return graph, nodes, edges


# Generate Candidate Locations
def generate_candidate_locations(df, center_lat, center_lon, radius_km, high_density_threshold=0.7, base_points=10):
    """
    Generate more candidate locations in high-density areas and at least one in low-density areas.
    """
    # Convert radius to degrees
    lat_radius = radius_km / 111
    lon_radius = radius_km / (111 * np.cos(np.radians(center_lat)))

    # Split the data into high-density and low-density
    high_density = df[df['Normalized_Density'] >= high_density_threshold]
    low_density = df[df['Normalized_Density'] < high_density_threshold]

    # Generate candidates for high-density areas
    high_density_candidates = pd.DataFrame([
        (lat, lon) for lat in np.linspace(center_lat - lat_radius, center_lat + lat_radius, base_points * 2)
        for lon in np.linspace(center_lon - lon_radius, center_lon + lon_radius, base_points * 2)
    ], columns=["Latitude", "Longitude"])

    # Ensure at least one candidate for each low-density location
    low_density_candidates = low_density[['Latitude', 'Longitude']].drop_duplicates()

    # Combine high- and low-density candidates
    all_candidates = pd.concat([high_density_candidates, low_density_candidates], ignore_index=True)
    return all_candidates


# Assign Density to Candidates
def assign_density_to_candidates(candidates, df):
    density_tree = cKDTree(df[['Latitude', 'Longitude']])
    distances, indices = density_tree.query(candidates[['Latitude', 'Longitude']], k=1)
    candidates['Density'] = df.iloc[indices]['Normalized_Density'].values
    return candidates


# Prepare Graph Data with Candidates
def prepare_graph_data_with_candidates(df, graph, nodes, candidates):
    combined_df = pd.concat([df, candidates], ignore_index=True)
    combined_df['Node_ID'] = combined_df.apply(
        lambda row: ox.distance.nearest_nodes(graph, row['Longitude'], row['Latitude']), axis=1
    )
    combined_df['Node_ID'] = combined_df['Node_ID'].astype(int)

    node_features = pd.DataFrame(index=nodes.index)
    node_features['Density'] = 0
    node_features.loc[combined_df['Node_ID'], 'Density'] = combined_df['Density'].astype(int).values

    valid_indices = list(node_features.index)
    filtered_edges = [(u, v) for u, v, *_ in graph.edges if u in valid_indices and v in valid_indices]

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(valid_indices)}
    reindexed_edges = [(node_id_map[u], node_id_map[v]) for u, v in filtered_edges]

    edge_index = torch.tensor(reindexed_edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features['Density'].fillna(0).values, dtype=torch.float).view(-1, 1)

    if edge_index.numel() > 0 and edge_index.max() >= x.size(0):
        raise ValueError(f"Edge indices exceed node features size: max index {edge_index.max()}, node size {x.size(0)}")

    data = Data(x=x, edge_index=edge_index)
    return data, combined_df


# Define GNN Model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


# Train GNN
def train_gnn(data, df, epochs=300, lr=0.01):
    model = GCN(input_dim=1, hidden_dim=16, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)

        # Custom loss: Higher weight for high-density areas, minimum weight for low-density areas
        weights = data.x.squeeze()
        min_weight = 0.1  # Ensure low-density areas have non-zero weight
        adjusted_weights = torch.where(weights > 0, weights + min_weight, torch.tensor(min_weight))

        # Weighted MSE loss
        loss = ((adjusted_weights * (out.squeeze() - data.x.squeeze()) ** 2).mean())
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model


# Predict Candidates
def predict_candidates(model, data, candidates, low_density_threshold=0.7):
    """
    Predict probabilities for candidate nodes and ensure at least one bus stop in low-density areas.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(data).squeeze().numpy()
        candidates['Prediction'] = predictions[-len(candidates):]
        candidates['Predicted_Stop'] = (candidates['Prediction'] > 0.5).astype(int)

        # Ensure at least one bus stop in low-density areas
        low_density_candidates = candidates[candidates['Density'] < low_density_threshold]
        if low_density_candidates['Predicted_Stop'].sum() == 0:
            # Select the candidate with the highest prediction in low-density areas
            idx = low_density_candidates['Prediction'].idxmax()
            candidates.loc[idx, 'Predicted_Stop'] = 1

    return candidates


# Adjust predictions to nearest road nodes
def adjust_predictions_to_road(candidates, graph):
    # Snap each candidate to the nearest node on the road network
    candidates['Adjusted_Node_ID'] = candidates.apply(
        lambda row: ox.distance.nearest_nodes(graph, X=row['Longitude'], Y=row['Latitude']), axis=1
    )

    # Map the snapped node IDs back to their coordinates in the graph
    candidates['Adjusted_Latitude'] = candidates['Adjusted_Node_ID'].map(
        lambda node_id: graph.nodes[node_id]['y'] if node_id in graph.nodes else np.nan
    )
    candidates['Adjusted_Longitude'] = candidates['Adjusted_Node_ID'].map(
        lambda node_id: graph.nodes[node_id]['x'] if node_id in graph.nodes else np.nan
    )

    # Drop candidates where snapping failed (e.g., outside the graph area)
    candidates = candidates.dropna(subset=['Adjusted_Latitude', 'Adjusted_Longitude']).reset_index(drop=True)
    return candidates


# Visualize Results (Updated for Adjusted Points)
def visualize_candidate_predictions(candidates, output_html):
    map_bamberg = folium.Map(location=[49.8988, 10.9028], zoom_start=13)
    for _, row in candidates.iterrows():
        color = 'blue' if row['Density'] >= 0.7 else 'red'
        folium.Marker(
            location=[row['Adjusted_Latitude'], row['Adjusted_Longitude']],
            popup=f"Adjusted Location - Prediction: {row['Prediction']:.2f}, Density: {row['Density']:.2f}",
            icon=folium.Icon(color=color, icon='ok-sign')
        ).add_to(map_bamberg)
    map_bamberg.save(output_html)


# Main Function (Updated Integration)
def main():
    input_file = "Training Data/final_busStop_density.ods"
    output_file = "Model Data/GNN-predicted_candidates.ods"
    output_html = "Template/GNN-candidate_predictions.html"

    # Load the density data
    df = load_data(input_file)

    # Load the road network graph
    graph, nodes, _ = get_osm_data()

    # Generate candidates based on density
    candidates = generate_candidate_locations(df, 49.8988, 10.9028, 5, high_density_threshold=0.7, base_points=10)

    # Assign density to the generated candidates
    candidates = assign_density_to_candidates(candidates, df)

    # Prepare graph data with candidates
    data, combined_df = prepare_graph_data_with_candidates(df, graph, nodes, candidates)

    # Train the GNN model
    model = train_gnn(data, df)

    # Predict probabilities for candidates
    candidates = predict_candidates(model, data, candidates)

    # Snap predicted locations to the road network
    candidates = adjust_predictions_to_road(candidates, graph)

    # Save and visualize results
    candidates.to_excel(output_file, engine='odf')
    visualize_candidate_predictions(candidates, output_html)
    print(f"Results saved to {output_file} and visualized in {output_html}")


if __name__ == "__main__":
    main()

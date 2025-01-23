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
def generate_candidate_locations(center_lat, center_lon, radius, num_points):
    lat_range = np.linspace(center_lat - radius, center_lat + radius, num_points)
    lon_range = np.linspace(center_lon - radius, center_lon + radius, num_points)
    candidates = [(lat, lon) for lat in lat_range for lon in lon_range]
    return pd.DataFrame(candidates, columns=["Latitude", "Longitude"])


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
def train_gnn(data, df, epochs=100, lr=0.01):
    model = GCN(input_dim=1, hidden_dim=16, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.x)
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
def adjust_predictions_to_road(df, graph):
    """
    Adjust predicted bus stop locations to the nearest road node using the OSM graph.
    """
    df['Adjusted_Node_ID'] = df.apply(
        lambda row: ox.distance.nearest_nodes(graph, row['Longitude'], row['Latitude']), axis=1
    )

    df['Adjusted_Latitude'] = df['Adjusted_Node_ID'].map(graph.nodes).apply(lambda x: x['y'])
    df['Adjusted_Longitude'] = df['Adjusted_Node_ID'].map(graph.nodes).apply(lambda x: x['x'])

    return df


# Visualize Results
def visualize_candidate_predictions(candidates, output_html):
    map_bamberg = folium.Map(location=[49.8988, 10.9028], zoom_start=13)
    for _, row in candidates.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Density: {row['Density']:.2f}, Prediction: {row['Prediction']:.2f}",
            icon=folium.Icon(color='blue' if row['Predicted_Stop'] else 'red')
        ).add_to(map_bamberg)
    map_bamberg.save(output_html)


# Main Function
def main():
    input_file = "Training Data/final_busStop_density.ods"
    output_file = "Model Data/GNN-predicted_candidates.ods"
    output_html = "Template/GNN-candidate_predictions.html"

    df = load_data(input_file)
    graph, nodes, _ = get_osm_data()

    candidates = generate_candidate_locations(49.8988, 10.9028, 0.01, 10)
    candidates = assign_density_to_candidates(candidates, df)

    data, combined_df = prepare_graph_data_with_candidates(df, graph, nodes, candidates)

    model = train_gnn(data, df)

    candidates = predict_candidates(model, data, candidates)

    candidates = adjust_predictions_to_road(candidates, graph)

    candidates.to_excel(output_file, engine='odf')
    visualize_candidate_predictions(candidates, output_html)
    print(f"Results saved to {output_file} and visualized in {output_html}")


if __name__ == "__main__":
    main()

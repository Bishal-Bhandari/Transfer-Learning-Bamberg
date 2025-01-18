import pandas as pd
import osmnx as ox
import folium
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_absolute_error, accuracy_score
import numpy as np


# Load ODS file and Normalize it 1
def load_data(file_path):
    df = pd.read_excel(file_path, engine='odf')
    scaler = MinMaxScaler()
    df['Normalized_Density'] = scaler.fit_transform(df[['Density']])
    return df


# Load OSM data 2
def get_osm_data():
    graph = ox.graph_from_place("Bamberg, Germany", network_type='drive')
    nodes, edges = ox.graph_to_gdfs(graph)
    return graph, nodes, edges


# Prepare data for GNN 3
def prepare_graph_data(df, graph, nodes):
    # Map lat/long to nearest OSM nodes
    df['Node_ID'] = df.apply(
        lambda row: ox.distance.nearest_nodes(graph, row['Longitude'], row['Latitude']), axis=1
    )

    # cast Node_ID to int
    df['Node_ID'] = df['Node_ID'].astype(int)

    # normalized density mapped to graph nodes
    node_features = pd.DataFrame(index=nodes.index)
    node_features['Density'] = 0

    # assign normalized density to corresponding node
    node_features.loc[df['Node_ID'], 'Density'] = df['Normalized_Density'].values.astype(int)

    # Filter to include  valid node
    valid_indices = list(node_features.index)  # Convert to list
    filtered_edges = [(u, v) for u, v, *_ in graph.edges if u in valid_indices and v in valid_indices]

    # Reindex edges to the valid node
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(valid_indices)}
    reindexed_edges = [(node_id_map[u], node_id_map[v]) for u, v in filtered_edges]

    # Create edge_index tensor
    edge_index = torch.tensor(reindexed_edges, dtype=torch.long).t().contiguous()

    # Filter node_features to match the graph nodes
    node_features = node_features.loc[valid_indices]

    # Create feature tensor
    x = torch.tensor(node_features['Density'].fillna(0).values, dtype=torch.float).view(-1, 1)

    # Validate edge_index range
    if edge_index.numel() > 0 and edge_index.max() >= x.size(0):
        raise ValueError(f"Edge indices exceed node features size: max index {edge_index.max()}, node size {x.size(0)}")

    # Return the PyTorch Geometric object
    data = Data(x=x, edge_index=edge_index)
    return data, df


# Define GNN Model 4
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


# Train and Predict 5
def train_gnn(data, df, actual_values, epochs=100, lr=0.01):
    model = GCN(input_dim=1, hidden_dim=16, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.x)
        loss.backward()
        optimizer.step()

        # Calculate MAE and accuracy
        predictions = out.squeeze().detach().numpy()

        df_predictions = df.set_index('Node_ID').reindex(df['Node_ID'])  # aligning 'Node_ID'
        aligned_actuals = df_predictions['Normalized_Density'].values
        aligned_predictions = predictions[:len(aligned_actuals)]  #  using the corresponding predictions

        # Calculate MAE and accuracy
        mae = mean_absolute_error(aligned_actuals, aligned_predictions)
        accuracy = accuracy_score((aligned_predictions > 0.5).astype(int), (aligned_actuals > 0.5).astype(int))

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, MAE: {mae:.4f}, Accuracy: {accuracy * 100:.2f}%")

    return model


def predict_gnn(model, data, df):
    model.eval()
    with torch.no_grad():
        predictions = model(data).squeeze().numpy()

        # Creating a series with the predictions and align it with df['Node_ID']
        node_predictions = pd.Series(predictions, index=range(len(predictions)))  # Use an integer range as the index
        node_predictions = node_predictions.reindex(df['Node_ID']).values  # Align predictions with df['Node_ID']

        # Setting the predictions as the 'Probability' column in df
        df['Probability'] = node_predictions
    return df


# Visualize Results 6
def visualize_results(df, output_html):
    map_bamberg = folium.Map(location=[49.8988, 10.9028], zoom_start=13)

    for _, row in df.iterrows():
        location_name = row.get('Location Name', 'Unknown Location')  # revert if 'Location Name' doesn't exist
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Location: {location_name}\nProbability: {row['Probability']:.2f}",
            icon=folium.Icon(color='blue' if row['Probability'] > 0.5 else 'red')
        ).add_to(map_bamberg)

    map_bamberg.save(output_html)


# Main Function 7
def main():
    # File paths
    input_file = "Training Data/final_busStop_density.ods"
    output_file = "Model Data/GNN-predicted_bus_stops.ods"
    output_html = "Template/GNN-bus_stop_predictions.html"

    # Load and preprocess data
    df = load_data(input_file)

    # Load OSM data
    graph, nodes, _ = get_osm_data()

    # Prepare graph data
    data, df = prepare_graph_data(df, graph, nodes)

    # values for MAE and accuracy calculation
    actual_values = df['Normalized_Density'].values

    # Train GNN model and calculate performance
    model = train_gnn(data, df, actual_values)

    # Predict probabilities
    df = predict_gnn(model, data, df)

    # Save results
    df.to_excel(output_file, engine='odf')

    # Visualize results
    visualize_results(df, output_html)
    print(f"Results saved to {output_file} and visualized in {output_html}")


if __name__ == "__main__":
    main()

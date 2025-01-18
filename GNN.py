import pandas as pd
import osmnx as ox
import folium
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# Step 1: Load ODS File and Normalize Data
def load_data(file_path):
    df = pd.read_excel(file_path, engine='odf')
    scaler = MinMaxScaler()
    df['Normalized_Density'] = scaler.fit_transform(df[['Density']])
    return df


# Step 2: Load OSM Data
def get_osm_data():
    graph = ox.graph_from_place("Bamberg, Germany", network_type='drive')
    nodes, edges = ox.graph_to_gdfs(graph)
    return graph, nodes, edges


# Step 3: Prepare Data for GNN
def prepare_graph_data(df, graph, nodes):
    # Map lat/long to nearest OSM nodes
    df['Node_ID'] = df.apply(
        lambda row: ox.distance.nearest_nodes(graph, row['Longitude'], row['Latitude']), axis=1
    )

    # Explicitly cast Node_ID to int
    df['Node_ID'] = df['Node_ID'].astype(int)

    # Create node features (normalized density mapped to graph nodes)
    node_features = pd.DataFrame(index=nodes.index)
    node_features['Density'] = 0

    # Assign normalized density to corresponding node IDs
    node_features.loc[df['Node_ID'], 'Density'] = df['Normalized_Density'].values

    # Ensure all indices in node_features match with graph nodes
    valid_indices = node_features.index

    # Filter edges to include only valid node indices
    edges = [(u, v) for u, v, *_ in graph.edges if u in valid_indices and v in valid_indices]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Ensure alignment of `node_features` with PyTorch tensors
    x = torch.tensor(node_features['Density'].fillna(0).values, dtype=torch.float).view(-1, 1)

    # Ensure `edge_index` is non-empty
    if edge_index.numel() == 0:
        raise ValueError("No valid edges found in the graph. Check the input data.")

    data = Data(x=x, edge_index=edge_index)
    return data, df




# Step 4: Define GNN Model
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


# Step 5: Train and Predict

def train_gnn(data, epochs=100, lr=0.01):
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    return model


def predict_gnn(model, data, df):
    model.eval()
    with torch.no_grad():
        predictions = model(data).squeeze().numpy()

        # Fix: Align predictions with Node_ID using .reindex()
        node_predictions = pd.Series(predictions, index=data.x.index).reindex(df['Node_ID']).values
        df['Probability'] = node_predictions
    return df


# Step 6: Visualize Results
def visualize_results(df, output_html):
    map_bamberg = folium.Map(location=[49.8988, 10.9028], zoom_start=13)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Location: {row['Location Name']}\nProbability: {row['Probability']:.2f}",
            icon=folium.Icon(color='blue' if row['Probability'] > 0.5 else 'red')
        ).add_to(map_bamberg)

    map_bamberg.save(output_html)


# Step 7: Main Function
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

    # Train GNN model
    model = train_gnn(data)

    # Predict probabilities
    df = predict_gnn(model, data, df)

    # Save results
    df.to_excel(output_file, engine='odf')

    # Visualize results
    visualize_results(df, output_html)
    print(f"Results saved to {output_file} and visualized in {output_html}")


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
from OSMPythonTools import data as osmp_data  # renamed to avoid conflict
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import joblib


# 1. Data Preparation
def prepare_data(stop_data):
    """
    Process raw bus stop data into graph format and generate labels using clustering.
    Expected columns in stop_data: 'direction', 'route_short_name', 'stop_id',
    'stop_lat', 'stop_lon', 'stop_sequence'.
    """
    # Convert categorical columns to numeric codes if needed
    for col in ['direction', 'route_short_name', 'stop_id']:
        if stop_data[col].dtype == 'object':
            stop_data[col] = stop_data[col].astype('category').cat.codes

    # Use all six columns as features
    features = stop_data[['direction', 'route_short_name', 'stop_id', 'stop_lat', 'stop_lon', 'stop_sequence']].values

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create graph edges based on spatial proximity in the scaled feature space
    nn_model = NearestNeighbors(n_neighbors=3).fit(features_scaled)
    distances, indices = nn_model.kneighbors(features_scaled)

    # Build edge_index (avoiding self-loops)
    edge_list = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Generate binary labels using KMeans clustering with 2 clusters.
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features_scaled)

    return features_scaled, edge_index, scaler, labels


# 2. GNN Model Architecture
class BusStopGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.predictor = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.predictor(x)


# 3. Training Pipeline
def train_model(features, edge_index, labels, num_epochs=1000):
    # Convert labels to float and reshape to (N, 1) for binary classification
    labels = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

    # Create PyG Data object
    graph_data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=labels
    )

    # Create train/validation masks (80/20 split)
    num_nodes = len(features)
    mask = torch.rand(num_nodes) < 0.8
    graph_data.train_mask = mask
    graph_data.val_mask = ~mask

    # Initialize model with input_dim=6, hidden_dim=64, output_dim=1
    model = BusStopGNN(
        input_dim=features.shape[1],
        hidden_dim=64,
        output_dim=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        # Use binary cross entropy with logits loss for binary classification
        loss = F.binary_cross_entropy_with_logits(out[graph_data.train_mask],
                                                  graph_data.y[graph_data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(graph_data)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds[graph_data.val_mask] == graph_data.y[graph_data.val_mask]).sum().item()
            acc = correct / graph_data.val_mask.sum().item()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'Output/best_bus_stop_model.pth')

        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}')

    return model, graph_data


# 4. Prediction Function
def predict_new_stop(new_coords, graph_data, model_path='Output/best_bus_stop_model.pth'):
    """
    Predict the binary label for a new bus stop using its features.
    new_coords: list of 6 elements corresponding to
    [direction, route_short_name, stop_id, stop_lat, stop_lon, stop_sequence]
    """
    # Load the trained model and scaler
    model = BusStopGNN(input_dim=6, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scaler = joblib.load('Output/bus_stop_scaler.pkl')

    # Preprocess new data
    scaled_coords = scaler.transform([new_coords])
    new_x = torch.tensor(scaled_coords, dtype=torch.float)

    # Append new stop features to the existing graph features
    full_features = torch.cat([graph_data.x, new_x])
    nn_model = NearestNeighbors(n_neighbors=3).fit(graph_data.x.numpy())
    _, indices = nn_model.kneighbors(new_x.numpy())

    new_edges = []
    new_node_idx = len(full_features) - 1
    for idx in indices[0]:
        new_edges.append([new_node_idx, idx])
        new_edges.append([idx, new_node_idx])
    new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous()

    pred_edge_index = torch.cat([graph_data.edge_index, new_edges_tensor], dim=1)
    pred_data = Data(x=full_features, edge_index=pred_edge_index)

    # Make prediction for the new node
    with torch.no_grad():
        logits = model(pred_data)
        new_node_logit = logits[new_node_idx]
        prob = torch.sigmoid(new_node_logit)
        predicted_label = int(prob > 0.5)

    return predicted_label, prob.item()


# 5. Main Execution Flow
if __name__ == "__main__":
    # Load your bus stop data
    bus_stops = pd.read_excel("Training Data/stib_stops.ods", engine='odf')

    # Prepare data using all six specified columns
    features, edge_index, scaler, labels = prepare_data(bus_stops)

    # Save the scaler for future predictions
    joblib.dump(scaler, 'Output/bus_stop_scaler.pkl')

    # Train the model and obtain graph data
    trained_model, graph_data = train_model(features, edge_index, labels)

    # Example prediction with a new bus stop feature vector (example values)
    # Replace these with actual values as needed.
    new_stop = [0, 1, 12345, 50.8503, 4.3517, 5]
    predicted_label, prediction_prob = predict_new_stop(new_stop, graph_data)
    print(f"Prediction probability: {prediction_prob}")
    print(f"Predicted binary label for the new stop: {predicted_label}")

# New imports
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch_geometric.utils import from_networkx
import torch.nn.functional as f

def create_node_features_and_labels(road_graph, bus_stops):
    """Create feature matrix and labels for all nodes in the graph"""
    features = []
    labels = []
    node_ids = []

    # Feature keys we want to use (excluding temporal features)
    feature_keys = ['poi_score', 'density_rank']

    for node in road_graph.nodes(data=True):
        node_id = node[0]
        data = node[1]

        # Create features
        node_features = [
            data.get('poi_score', 0),
            data.get('density_rank', 0),
        ]

        # Add coordinates
        if 'x' in data and 'y' in data:
            node_features += [data['x'], data['y']]
        else:
            node_features += [0.0, 0.0]

        features.append(node_features)

        # Create labels (1 for bus stops, 0 for others)
        labels.append(1 if 'stop_name' in data else 0)
        node_ids.append(node_id)

    return np.array(features), np.array(labels), node_ids


class BusStopPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BusStopPredictor, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = f.relu(self.conv1(x, edge_index))
        x = f.dropout(x, p=0.5, training=self.training)
        x = f.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return f.log_softmax(x, dim=1)


def train_and_save_model(bus_stops, road_graph, output_path="bus_stop_predictor.pth"):
    # Create feature matrix and labels
    features, labels, node_ids = create_node_features_and_labels(road_graph, bus_stops)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Convert to PyTorch tensors
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.LongTensor(labels)

    # Convert NetworkX graph to PyG Data
    pyg_data = from_networkx(road_graph)
    pyg_data.x = features_tensor
    pyg_data.y = labels_tensor

    # Create train/test masks
    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Split data (80% train, 20% test)
    indices = torch.randperm(num_nodes)
    train_indices = indices[:int(0.8 * num_nodes)]
    test_indices = indices[int(0.8 * num_nodes):]

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    pyg_data.train_mask = train_mask
    pyg_data.test_mask = test_mask

    # Model parameters
    input_dim = features.shape[1]
    hidden_dim = 32
    output_dim = 2  # Binary classification

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BusStopPredictor(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    pyg_data = pyg_data.to(device)

    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(pyg_data)
        loss = f.nll_loss(output[pyg_data.train_mask], pyg_data.y[pyg_data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Save model and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state': scaler,
        'node_ids': node_ids
    }, output_path)

    print(f"Model saved to {output_path}")
    return model

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.distance import great_circle
import joblib


# --------------------------
# 1. Load Bus Stop Data
# --------------------------
def load_bus_stops(ods_path):
    df = pd.read_excel(ods_path, engine='odf')
    print(f"Loaded {len(df)} bus stops")
    return df[['stop_name', 'stop_lat', 'stop_lon']]


# --------------------------
# 2. Road Network Integration
# --------------------------
def get_road_network(stops, city_name):
    # Download road network with modern parameters
    G = ox.graph_from_place(
        city_name,
        network_type="drive",
        simplify=False,
        truncate_by_edge=True
    )

    # Project to UTM
    G_proj = ox.project_graph(G)

    # Integrate stops using updated method
    for _, stop in stops.iterrows():
        point = (stop['stop_lat'], stop['stop_lon'])
        nearest_edge = ox.distance.nearest_edges(G_proj, stops['stop_lon'], stops['stop_lat'])

        # Use correct edge insertion syntax
        G_proj = ox.utils_graph.insert_node_along_edge(
            G_proj,
            nearest_edge,
            new_node_id=f"stop_{stop['stop_name']}",
            point=point
        )

    return G_proj


# --------------------------
# 3. Feature Engineering
# --------------------------
def create_features(road_network):
    features = []
    road_types = []

    # Get road type for each bus stop
    for node, data in road_network.nodes(data=True):
        if 'is_stop' in data:
            # Get connected edge properties
            edges = list(road_network.edges(node, data=True))
            if edges:
                road_type = edges[0][-1].get('highway', 'unknown')
                if isinstance(road_type, list):
                    road_type = road_type[0]
            else:
                road_type = 'unknown'

            features.append([
                data['y'],  # lat
                data['x'],  # lon
                road_type
            ])

    # Encode road types
    le = LabelEncoder()
    df = pd.DataFrame(features, columns=['lat', 'lon', 'road_type'])
    df['road_type_encoded'] = le.fit_transform(df['road_type'])

    # Normalize coordinates
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['lat', 'lon', 'road_type_encoded']])

    return scaled_features, scaler, le


# --------------------------
# 4. Graph Construction
# --------------------------
def build_bus_graph(road_network, features):
    G = nx.Graph()

    # Add nodes with features
    stops = [n for n, d in road_network.nodes(data=True) if 'is_stop' in d]
    for i, node in enumerate(stops):
        G.add_node(node, features=features[i])

    # Connect stops using multiple strategies
    for i, node1 in enumerate(stops):
        for j, node2 in enumerate(stops[i + 1:]):
            try:
                # Strategy 1: Use road network path
                path = nx.shortest_path(road_network, node1, node2, weight='length')
                if len(path) <= 3:
                    G.add_edge(node1, node2, weight=1 / (len(path) - 1))
            except nx.NetworkXNoPath:
                try:
                    # Strategy 2: Use geographic distance as fallback
                    coord1 = (road_network.nodes[node1]['y'], road_network.nodes[node1]['x'])
                    coord2 = (road_network.nodes[node2]['y'], road_network.nodes[node2]['x'])
                    distance = great_circle(coord1, coord2).meters

                    # Connect if within 1.5km and same road type
                    if distance < 1500:
                        road_type1 = road_network.nodes[node1].get('highway', 'unknown')
                        road_type2 = road_network.nodes[node2].get('highway', 'unknown')
                        if road_type1 == road_type2:
                            G.add_edge(node1, node2, weight=1 / (distance + 1e-5))
                except KeyError:
                    # Strategy 3: Add weak connection for complete graph
                    G.add_edge(node1, node2, weight=0.1)
                    continue

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


# --------------------------
# 5. GNN Model
# --------------------------
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


# --------------------------
# 6. Training Pipeline
# --------------------------
def train_model(graph, num_classes=2, epochs=200):
    # Convert to PyG Data
    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
    x = torch.tensor([graph.nodes[n]['features'] for n in graph.nodes], dtype=torch.float)

    # Create synthetic labels (example: predict if major transportation hub)
    labels = torch.tensor([int("station" in n.lower()) for n in graph.nodes], dtype=torch.long)

    data = Data(x=x, y=labels, edge_index=edge_index)
    data.train_mask = torch.zeros(len(x), dtype=torch.bool).bernoulli(0.8)
    data.val_mask = ~data.train_mask

    # Initialize model
    model = BusStopGNN(input_dim=x.shape[1],
                       hidden_dim=32,
                       output_dim=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Training loop
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(dim=1)
            acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')

        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}')

    return model


# --------------------------
# 7. Main Execution
# --------------------------
def main(ods_path, city_name):
    # 1. Load data
    stops = load_bus_stops(ods_path)

    # 2. Get integrated road network
    road_network = get_road_network(stops, city_name)

    # 3. Create features
    features, scaler, le = create_features(road_network)

    # 4. Build bus stop graph
    bus_graph = build_bus_graph(road_network, features)

    # 5. Train model
    model = train_model(bus_graph)

    # 6. Save artifacts
    joblib.dump(scaler, 'bus_scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    nx.write_gpickle(bus_graph, 'bus_graph.gpickle')
    print("\nTraining complete. Saved:")
    print("- Trained model (best_model.pth)")
    print("- Feature scaler (bus_scaler.pkl)")
    print("- Label encoder (label_encoder.pkl)")
    print("- Bus stop graph (bus_graph.gpickle)")


if __name__ == "__main__":
    # Example usage
    main('Training Data/stib_stops.ods', 'Brussels, Belgium')
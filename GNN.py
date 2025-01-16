import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from geopy.distance import geodesic
import folium

# Load data
data = pd.read_excel("Training Data/final_busStop_density.ods", engine='odf')

latitude = data['Latitude']
longitude = data['Longitude']
density = data['Density']

# Nearby roads feature
def get_nearby_roads(lat, lon, distance=500):
    G = ox.graph_from_point((lat, lon), dist=distance, network_type='all')
    return len(list(G.edges))  # Number of road segments

# Add nearby roads feature to data
data['Nearby_Roads'] = [get_nearby_roads(lat, lon) for lat, lon in zip(latitude, longitude)]

# Prepare features and labels
X = np.column_stack([latitude, longitude, density, data['Nearby_Roads']])
y = np.array([1 if density >= 3 else 0 for density in data['Density']])  # 1 for likely bus stop, 0 otherwise

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a graph
G = nx.Graph()

# Add nodes with features
for i, row in data.iterrows():
    G.add_node(i, lat=row['Latitude'], lon=row['Longitude'], density=row['Density'], nearby_roads=row['Nearby_Roads'])

# Define edges with proper distance calculation
threshold_distance = 1000  # meters
for i, node1 in data.iterrows():
    for j, node2 in data.iterrows():
        if i != j:
            lat1, lon1 = node1['Latitude'], node1['Longitude']
            lat2, lon2 = node2['Latitude'], node2['Longitude']
            dist = geodesic((lat1, lon1), (lat2, lon2)).meters
            if dist <= threshold_distance:
                G.add_edge(i, j)

# Data for PyTorch Geometric
node_features = torch.tensor(X_scaled, dtype=torch.float)
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()  # Edge index format

# Labels for nodes
labels = torch.tensor(y, dtype=torch.float)

# Create a data object for PyTorch Geometric
data_gnn = Data(x=node_features, edge_index=edge_index, y=labels)

# Graph Convolutional Network Model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.6)  # Add dropout for regularization

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index)
        return x

# Initialize the model
model = GNNModel(input_dim=X_scaled.shape[1], hidden_dim=64, output_dim=1)

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(data_gnn)

    # Loss
    loss = criterion(output.view(-1), data_gnn.y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/100, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    output = model(data_gnn)
    predictions = torch.sigmoid(output).round()  # Convert to 0 or 1
    accuracy = (predictions == data_gnn.y).float().mean()
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')

# Post-process predictions
predictions = torch.sigmoid(output).numpy().flatten()
data['Predicted_Bus_Stop'] = predictions

# Save results to ODS
data.to_excel("predicted_bus_stops.ods", engine='odf')

# Visualize on Folium map
city_lat, city_lon = 49.89, 10.89  # Bamberg center coordinates
city_map = folium.Map(location=[city_lat, city_lon], zoom_start=12)

# Add markers for predicted bus stops
for i, row in data.iterrows():
    if row['Predicted_Bus_Stop'] >= 0.5:
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=f"Density: {row['Density']}, Nearby Roads: {row['Nearby_Roads']}, Probability: {row['Predicted_Bus_Stop']:.2f}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(city_map)

# Save map as an HTML file
city_map.save('Template/predicted_bus_stops_map.html')
print("Map with predicted bus stops saved as.")

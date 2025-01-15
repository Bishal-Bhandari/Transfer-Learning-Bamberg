import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import folium

# Load data
data = pd.read_excel("Training Data/final_busStop_density.ods")

latitude = data['Latitude']
longitude = data['Longitude']
density = data['Density']


# nearby roads
def get_nearby_roads(lat, lon, distance=500):
    G = ox.graph_from_point((lat, lon), dist=distance, network_type='all')
    return len(list(G.edges))  # no of road segments


# add roads feature to data
data['Nearby_Roads'] = [get_nearby_roads(lat, lon) for lat, lon in zip(latitude, longitude)]

# Prepare features
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

# Define edges
threshold_distance = 1000
for i, node1 in enumerate(data.iterrows()):
    for j, node2 in enumerate(data.iterrows()):
        if i != j:
            lat1, lon1 = node1[1]['Latitude'], node1[1]['Longitude']
            lat2, lon2 = node2[1]['Latitude'], node2[1]['Longitude']
            dist = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
            if dist <= threshold_distance:
                G.add_edge(i, j)

# data for pytorch geometric
node_features = torch.tensor(X_scaled, dtype=torch.float)
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()  # Edge index format

# labels for nodes
labels = torch.tensor(y, dtype=torch.float)

# data object for pytorch geometric
data = Data(x=node_features, edge_index=edge_index, y=labels)


# Graph Convolutional Networks
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


# starting the model
model = GNNModel(input_dim=X_scaled.shape[1], hidden_dim=64, output_dim=1)

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(data)

    # Loss
    loss = criterion(output.view(-1), data.y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{100}, Loss: {loss.item()}')

# predictions
model.eval()
with torch.no_grad():
    output = model(data)
    predictions = torch.sigmoid(output).round()  # Convert to 0 or 1
    accuracy = (predictions == data.y).float().mean()
    print(f'Accuracy: {accuracy.item() * 100}%')

# Filter locations
predicted_bus_stops = predictions.detach().numpy().flatten()
print(predicted_bus_stops)
predicted_bus_stops = (predicted_bus_stops * density.values / 5).round()
bus_stop_lat_lon = data.x[predicted_bus_stops >= 0.5, :2]  # latitude and longitude for predicted bus stops

# Folium map
city_lat, city_lon = 49.89, 10.89  # Example: Bamberg latitude and longitude
city_map = folium.Map(location=[city_lat, city_lon], zoom_start=12)

# add bus stop markers
for lat, lon in bus_stop_lat_lon:
    folium.Marker([lat, lon], popup="Predicted Bus Stop").add_to(city_map)

# Save HTML file
city_map.save('Template/predicted_bus_stops_map.html')

print("Map with predicted bus stops saved as.")

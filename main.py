import folium
import osmnx as ox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import Adam
from geopy.geocoders import Nominatim

# Load your ODS data
data = pd.read_excel("Training Data/final_busStop_density.ods")


# Function to get latitude and longitude for a city
def get_city_latlon(city_name):
    geolocator = Nominatim(user_agent="bus_stop_prediction")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError("City not found.")


# Get city input
city_name = "Bamberg"
# city_name = input("Enter the city name for bus stop generation: ")
city_lat, city_lon = get_city_latlon(city_name)

# In the ODS
latitude = data['Latitude']
longitude = data['Longitude']
density = data['Density']


# fetch nearby POI
def get_nearby_roads(lat, lon, distance=500):
    # Get nearby streets
    G = ox.graph_from_point((lat, lon), dist=distance, network_type='all')
    return len(list(G.edges))  # Number of road segments nearby


data['Nearby_Roads'] = [get_nearby_roads(lat, lon) for lat, lon in zip(latitude, longitude)]

# Prepare the feature matrix (X) and labels (y)
X = np.column_stack([latitude, longitude, density, data['Nearby_Roads']])
y = np.array([1 if density >= 3 else 0 for density in data['Density']])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Define the neural network
class BusStopPredictionModel(nn.Module):
    def __init__(self):
        super(BusStopPredictionModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Instantiate and train the model
model = BusStopPredictionModel()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{100}, Loss: {loss.item()}')

# Model evaluation
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    output = model(X_test_tensor)
    predictions = output.round()  # Predict 0 or 1 based on probability
    accuracy = (predictions == y_test_tensor).float().mean()
    print(f'Accuracy: {accuracy.item() * 100}%')

# Now predict bus stops for all locations
predictions = model(torch.tensor(X_scaled, dtype=torch.float32))
predicted_bus_stops = predictions.detach().numpy().flatten()

# Filter locations where the model predicts a bus stop (1)
bus_stop_lat_lon = data[predicted_bus_stops >= 0.5][['Latitude', 'Longitude']]

# Create a Folium map centered around the city location
city_map = folium.Map(location=[city_lat, city_lon], zoom_start=12)

# Add markers for predicted bus stop locations
for _, row in bus_stop_lat_lon.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup="Predicted Bus Stop").add_to(city_map)

# Save the map to an HTML file
city_map.save(f'Template/{city_name}_bus_stops.html')

print(f"Map has been saved as {city_name}_bus_stops.html")

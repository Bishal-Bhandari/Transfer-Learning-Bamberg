import osmnx as ox
import networkx as nx
import pandas as pd
import folium
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

# --------------------------
# STEP 1: Load OSM Map Data
# --------------------------
# Define the area of interest (change this to your city or region)
city_name = "Bamberg, Germany"

# Load the road network from OSM
G = ox.graph_from_place(city_name, network_type="drive")

# Convert the graph to a GeoDataFrame for easier analysis
nodes, edges = ox.graph_to_gdfs(G)

# --------------------------
# STEP 2: Extract Features from OSM
# --------------------------
# Extract all road geometries
road_geometries = edges["geometry"]

# If bus route data is available in OSM, extract it (optional)
#bus_routes = ox.features_from_tags(G, {"route": "bus"})

# Generate sample points along the roads as potential bus stop candidates
candidate_stops = []
for geom in road_geometries:
    if isinstance(geom, LineString):
        candidate_stops.extend(list(geom.coords))

candidate_stops = pd.DataFrame(candidate_stops, columns=["Longitude", "Latitude"])

# --------------------------
# STEP 3: Prepare Input Data
# --------------------------
# Assume additional datasets like population density and POI exist
population_density_data = pd.read_excel("Training Data/final_busStop_density.ods", engine="odf")
poi_data = pd.read_excel("Training Data/osm_poi_rank_data.ods", engine="odf")

# Normalize density and popularity scores
scaler = MinMaxScaler()
population_density_data["density_normalized"] = scaler.fit_transform(
    population_density_data[["Density"]]
)
poi_data["popularity_normalized"] = scaler.fit_transform(
    poi_data[["popularity_rank"]]
)


# Add proximity to POI for each candidate stop
def nearest_poi(lat, lon):
    stop_point = Point(lon, lat)
    poi_points = poi_data.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    distances = [stop_point.distance(poi) for poi in poi_points]
    return min(distances) if distances else np.nan


candidate_stops["nearest_poi_dist"] = candidate_stops.apply(
    lambda row: nearest_poi(row["Latitude"], row["Longitude"]), axis=1
)

# Combine all features for training
features = candidate_stops[["nearest_poi_dist"]]
labels = candidate_stops[["Latitude", "Longitude"]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Save the training and testing data
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("Training and testing data saved to 'train_data.csv' and 'test_data.csv'")

# --------------------------
# (Optional) Reload Data in Future
# --------------------------
# Uncomment below to reload the saved data
# train_data = pd.read_csv("train_data.csv")
# test_data = pd.read_csv("test_data.csv")
# X_train = train_data.iloc[:, :-2]
# y_train = train_data.iloc[:, -2:]
# X_test = test_data.iloc[:, :-2]
# y_test = test_data.iloc[:, -2:]


# --------------------------
# STEP 4: Train the ML Model
# --------------------------
# Combine all features for training
features = candidate_stops[["nearest_poi_dist"]]
labels = candidate_stops[["Latitude", "Longitude"]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Define the neural network
model = models.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(2, activation="linear"),  # Predict Latitude and Longitude
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16)

# --------------------------
# STEP 5: Predict New Bus Stops
# --------------------------
# Predict stops for high-density areas
new_data = pd.DataFrame({"nearest_poi_dist": [0.1, 0.2]})  # Adjust input features
predicted_stops = model.predict(new_data)

# --------------------------
# STEP 6: Visualize on Map
# --------------------------
# Create a map centered on the city
city_map = folium.Map(location=[49.8930, 10.9028], zoom_start=13)  # Bamberg's center

# Add existing candidate stops to the map
for _, row in candidate_stops.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        tooltip="Candidate Stop",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(city_map)

# Add predicted bus stops
for stop in predicted_stops:
    folium.Marker(
        location=[stop[0], stop[1]],
        tooltip="Predicted Stop",
        icon=folium.Icon(color="green", icon="info-sign"),
    ).add_to(city_map)

# Save the map
city_map.save("predicted_bus_stops_with_osm.html")
print("Map saved as 'predicted_bus_stops_with_osm.html'")

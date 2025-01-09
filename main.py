import pandas as pd
import geopandas as gpd
import folium
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

# Load ODS files (change paths to your actual file locations)
bus_stop_data = pd.read_excel('Data/filtered_busStop_LatLong.ods', engine='odf')
population_density_data = pd.read_excel('Data/final_busStop_density.ods', engine='odf')
poi_data = pd.read_excel('Data/osm_poi_rank_data.ods', engine='odf')

# Merge population density with bus stop data
data = pd.merge(bus_stop_data, population_density_data, on="name", how="left")

# Convert POI lat-long to GeoDataFrame
poi_gdf = gpd.GeoDataFrame(
    poi_data, geometry=gpd.points_from_xy(poi_data.Longitude, poi_data.Latitude)
)

# Normalize POI popularity score and density
scaler = MinMaxScaler()
data['density_normalized'] = scaler.fit_transform(data[['density']])
poi_data['popularity_normalized'] = scaler.fit_transform(poi_data[['popularity_score']])


# --------------------------
# STEP 2: Feature Engineering
# --------------------------
# Example: Calculate distance from each bus stop to nearest POI


def nearest_poi(lat, lon):
    stop_point = Point(lon, lat)
    nearest = poi_gdf.geometry.apply(lambda x: stop_point.distance(x))
    return nearest.min()


data['nearest_poi_dist'] = data.apply(lambda row: nearest_poi(row.Latitude, row.Longitude), axis=1)

# Combine features into a dataset
features = data[['density_normalized', 'nearest_poi_dist']]
labels = data[['Latitude', 'Longitude']]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# --------------------------
# STEP 3: Build and Train the Model
# --------------------------
# Define a simple neural network
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='linear')  # Output latitude and longitude
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16)

# --------------------------
# STEP 4: Predict New Bus Stops
# --------------------------
# Example input: high-density areas with no bus stops
new_data = pd.DataFrame({
    'density_normalized': [0.9, 0.8],
    'nearest_poi_dist': [0.1, 0.2]
})

predicted_stops = model.predict(new_data)

# --------------------------
# STEP 5: Visualize Results
# --------------------------
# Create a map centered on the city
map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
city_map = folium.Map(location=map_center, zoom_start=13)

# Plot existing bus stops
for _, row in data.iterrows():
    folium.Marker(location=[row['Latitude'], row['Longitude']], tooltip="Existing Bus Stop").add_to(city_map)

# Plot predicted bus stops
for stop in predicted_stops:
    folium.Marker(location=[stop[0], stop[1]], tooltip="New Bus Stop", icon=folium.Icon(color='green')).add_to(city_map)

# Save the map
city_map.save("predicted_bus_stops.html")

print("Map saved as 'predicted_bus_stops.html'")

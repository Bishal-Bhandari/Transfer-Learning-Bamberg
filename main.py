import osmnx as ox
import networkx as nx
import pandas as pd
import folium
import requests
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

#Load OSM Map Data
city_name = "Bamberg, Germany"
G = ox.graph_from_place(city_name, network_type="drive")
nodes, edges = ox.graph_to_gdfs(G)

# Extract Features from OSM
road_geometries = edges["geometry"]

candidate_stops = []
for geom in road_geometries:
    if isinstance(geom, LineString):
        candidate_stops.extend(list(geom.coords))

candidate_stops = pd.DataFrame(candidate_stops, columns=["Longitude", "Latitude"])

# Fetch Weather Data
API_KEY = "your_openweathermap_api_key"  # Replace with your API key
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"


def fetch_weather(lat, lon):
    """Fetch weather data for given latitude and longitude."""
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"  # Use metric system for temperature, etc.
    }
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code == 200:
        weather = response.json()
        return {
            "temp": weather["main"]["temp"],
            "humidity": weather["main"]["humidity"],
            "wind_speed": weather["wind"]["speed"]
        }
    else:
        return {"temp": np.nan, "humidity": np.nan, "wind_speed": np.nan}


# Add weather data to candidate stops
weather_data = []
for _, row in candidate_stops.iterrows():
    weather = fetch_weather(row["Latitude"], row["Longitude"])
    weather_data.append(weather)

# Convert weather data to DataFrame
weather_df = pd.DataFrame(weather_data)

# Merge weather data with candidate stops
candidate_stops = pd.concat([candidate_stops, weather_df], axis=1)

# Load and Prepare Additional Data (Population Density, POI)
population_density_data = pd.read_excel("Training Data/final_busStop_density.ods", engine="odf")
poi_data = pd.read_excel("Training Data/osm_poi_rank_data.ods", engine="odf")

# Normalize density and popularity scores
scaler = MinMaxScaler()
population_density_data["density_normalized"] = scaler.fit_transform(population_density_data[["Density"]])
poi_data["popularity_normalized"] = scaler.fit_transform(poi_data[["popularity_rank"]])


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

# Normalize weather features
candidate_stops[["temp_normalized", "humidity_normalized", "wind_speed_normalized"]] = scaler.fit_transform(
    candidate_stops[["temp", "humidity", "wind_speed"]]
)

# Combine Features for Model Training
features = candidate_stops[["nearest_poi_dist", "temp_normalized", "humidity_normalized", "wind_speed_normalized"]]
labels = candidate_stops[["Latitude", "Longitude"]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Save the training and testing data
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv("Model Data/train_data.csv", index=False)
test_data.to_csv("Model Data/test_data.csv", index=False)

print("Training and testing data saved to 'train_data.csv' and 'test_data.csv'")

# STEP 6: Define and Train ML Model
model = models.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(2, activation="linear"),  # Predict Latitude and Longitude
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16)

# Predict New Bus Stops
# Example of predicting new bus stops (adjust features as necessary)
new_data = pd.DataFrame({
    "nearest_poi_dist": [0.1, 0.2],
    "temp_normalized": [0.5, 0.6],
    "humidity_normalized": [0.7, 0.8],
    "wind_speed_normalized": [0.3, 0.4]
})

predicted_stops = model.predict(new_data)

# Visualize Results on Map
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
city_map.save("predicted_bus_stops_with_weather_and_osm.html")
print("Map saved as 'predicted_bus_stops_with_weather_and_osm.html'")

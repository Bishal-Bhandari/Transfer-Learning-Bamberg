import json
import os
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
from shapely.geometry import LineString
from scipy.spatial import cKDTree

# Set global paths
BASE_DIR = "Training Data/"
SAVE_DIR = "Model Data/"
CANDIDATE_STOPS_FILE = os.path.join(BASE_DIR, "filtered_busStop_LatLong.ods")
POP_DENSITY_FILE = os.path.join(BASE_DIR, "final_busStop_density.ods")
POI_RANK_FILE = os.path.join(BASE_DIR, "osm_poi_rank_data.ods")
API_KEYS_FILE = "api_keys.json"  # Update if needed


# Load OSM Map Data
def load_osm_data(city_name, network_type="drive"):
    G = ox.graph_from_place(city_name, network_type=network_type)
    _, edges = ox.graph_to_gdfs(G)
    road_geometries = edges["geometry"]
    candidate_stops = []
    for geom in road_geometries:
        if isinstance(geom, LineString):
            candidate_stops.extend(list(geom.coords))
    return pd.DataFrame(candidate_stops, columns=["Longitude", "Latitude"])


# Fetch Weather Data
def load_api_keys(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


def fetch_weather(lat, lon, api_key, url="http://api.openweathermap.org/data/2.5/weather"):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        weather = response.json()
        rain = weather.get("rain", {}).get("1h", np.nan)
        return {
            "temp": weather["main"]["temp"],
            "humidity": weather["main"]["humidity"],
            "wind_speed": weather["wind"]["speed"],
            "rain": rain
        }
    else:
        return {"temp": np.nan, "humidity": np.nan, "wind_speed": np.nan, "rain": np.nan}


def add_weather_data(candidate_stops, api_key):
    weather_data = [
        fetch_weather(row["Latitude"], row["Longitude"], api_key)
        for _, row in candidate_stops.iterrows()
    ]
    weather_df = pd.DataFrame(weather_data)
    return pd.concat([candidate_stops, weather_df], axis=1)


# Normalize Data
def normalize_data(df, columns):
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df[columns])
    return pd.DataFrame(normalized, columns=[col + "_normalized" for col in columns], index=df.index)


# Compute Proximity to POI
def compute_nearest_poi(candidate_stops, poi_data):
    candidate_coords = candidate_stops[["Longitude", "Latitude"]].to_numpy()
    poi_coords = poi_data[["Longitude", "Latitude"]].to_numpy()
    poi_tree = cKDTree(poi_coords)
    distances, _ = poi_tree.query(candidate_coords)
    candidate_stops["nearest_poi_dist"] = distances
    return candidate_stops


# Assign Population Density to Stops
def assign_population_density(candidate_stops, population_density_data):
    density_coords = population_density_data[["Longitude", "Latitude"]].to_numpy()
    density_values = population_density_data["Density_Rank"].to_numpy()
    density_tree = cKDTree(density_coords)
    distances, indices = density_tree.query(candidate_stops[["Longitude", "Latitude"]].to_numpy())
    candidate_stops["population_density"] = density_values[indices]
    return candidate_stops


# Prepare Data
def prepare_data(candidate_stops, features_columns, labels_columns, test_size=0.2):
    features = candidate_stops[features_columns]
    labels = candidate_stops[labels_columns]
    return train_test_split(features, labels, test_size=test_size)


# Build and Train Model
def build_and_train_model(X_train, y_train, input_shape):
    model = models.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, validation_split=0.2, epochs=25, batch_size=16)
    return model


# Main Execution
def main():
    # Load data
    api_keys = load_api_keys(API_KEYS_FILE)
    population_density_data = pd.read_excel(POP_DENSITY_FILE, engine="odf")
    poi_data = pd.read_excel(POI_RANK_FILE, engine="odf")

    # Preprocess candidate stops
    candidate_stops = pd.read_excel(CANDIDATE_STOPS_FILE, engine="odf")
    candidate_stops = add_weather_data(candidate_stops, api_keys['Weather_API']['API_key'])
    candidate_stops = assign_population_density(candidate_stops, population_density_data)
    candidate_stops = compute_nearest_poi(candidate_stops, poi_data)

    # Normalize and prepare data
    features_columns = ["nearest_poi_dist", "temp", "humidity", "wind_speed", "population_density", "rain"]
    labels_columns = ["Latitude", "Longitude"]
    X_train, X_test, y_train, y_test = prepare_data(candidate_stops, features_columns, labels_columns)

    # Train model
    model = build_and_train_model(X_train, y_train, len(features_columns))

    # Save and predict
    output_dir = os.path.join(SAVE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    candidate_stops.to_excel(os.path.join(output_dir, "processed_candidate_stops.ods"), engine="odf")


if __name__ == "__main__":
    main()

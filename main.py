import json
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
from scipy.spatial import cKDTree


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
def load_api_keys(file_path='api_keys.json'):
    with open(file_path) as json_file:
        return json.load(json_file)


def clean_poi_data(poi_data):
    # Check for missing or infinite values
    poi_data = poi_data.dropna(subset=["Longitude", "Latitude"])  # Drop rows with NaN values
    poi_data = poi_data[
        np.isfinite(poi_data[["Longitude", "Latitude"]]).all(axis=1)]  # Remove rows with infinite values
    return poi_data


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
        rain = weather.get("rain", {}).get("1h", np.nan)  # Get rain data if available
        return {
            "temp": weather["main"]["temp"],
            "humidity": weather["main"]["humidity"],
            "wind_speed": weather["wind"]["speed"],
            "rain": rain  # Add rain data
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


# Normalize data
def normalize_data(df, columns):
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df[columns])
    return pd.DataFrame(normalized, columns=[col + "_normalized" for col in columns], index=df.index)


# Compute proximity to POI
def compute_nearest_poi(candidate_stops, poi_data):
    candidate_coords = candidate_stops[["Longitude", "Latitude"]].to_numpy()
    poi_coords = poi_data[["Longitude", "Latitude"]].to_numpy()
    if not np.isfinite(poi_coords).all():
        raise ValueError("POI data contains NaN or infinite values. Please clean the data.")
    poi_tree = cKDTree(poi_coords)
    distances, _ = poi_tree.query(candidate_coords)
    candidate_stops["nearest_poi_dist"] = distances
    return candidate_stops


# Assign population density to candidate stops
def assign_population_density(candidate_stops, population_density_data):
    density_coords = population_density_data[["Longitude", "Latitude"]].to_numpy()
    density_values = population_density_data["Density_Rank"].to_numpy()
    density_tree = cKDTree(density_coords)
    distances, indices = density_tree.query(candidate_stops[["Longitude", "Latitude"]].to_numpy())
    candidate_stops["population_density"] = density_values[indices]
    return candidate_stops


# Prepare training and testing data
def prepare_data(candidate_stops, features_columns, labels_columns, test_size=0.2):
    features = candidate_stops[features_columns]
    labels = candidate_stops[labels_columns]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test


# Build and train the model
def build_and_train_model(X_train, y_train, input_shape):
    model = models.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16)
    return model


# Visualize results on map
def visualize_results_on_map(candidate_stops, predicted_stops, output_file, city_center, zoom_start=13):
    city_map = folium.Map(location=city_center, zoom_start=zoom_start)
    for _, row in candidate_stops.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            tooltip="Candidate Stop",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(city_map)
    for stop in predicted_stops:
        folium.Marker(
            location=[stop[0], stop[1]],
            tooltip="Predicted Stop",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(city_map)
    city_map.save(output_file)


# Save data in ODS format
def save_data_as_ods(data, file_name):
    data.to_excel(file_name, engine="odf", index=False)


def main():
    # Main process
    city_name = "Bamberg, Germany"
    candidate_stops = load_osm_data(city_name)
    api_keys = load_api_keys()
    candidate_stops = add_weather_data(candidate_stops, api_keys['Weather_API']['API_key'])

    # Load population density data
    population_density_data = pd.read_excel("Training Data/final_busStop_density.ods", engine="odf")
    population_density_data = population_density_data.rename(
        columns={"Longitude": "Longitude", "Latitude": "Latitude", "Density": "Density_Rank"})

    # Assign population density to candidate stops
    candidate_stops = assign_population_density(candidate_stops, population_density_data)
    candidate_stops = pd.concat(
        [candidate_stops,
         normalize_data(candidate_stops, ["temp", "humidity", "wind_speed", "rain" "population_density"])],
        axis=1)

    # Load and process POI data
    poi_data = pd.read_excel("Training Data/osm_poi_rank_data.ods", engine="odf")
    poi_data = clean_poi_data(poi_data)
    poi_data["popularity_normalized"] = normalize_data(poi_data, ["popularity_rank"])["popularity_rank_normalized"]
    candidate_stops = compute_nearest_poi(candidate_stops, poi_data)

    # Prepare data for model training
    features_columns = [
        "nearest_poi_dist", "temp_normalized", "humidity_normalized", "wind_speed_normalized",
        "population_density_normalized", "rain"]

    labels_columns = ["Latitude", "Longitude"]
    X_train, X_test, y_train, y_test = prepare_data(candidate_stops, features_columns, labels_columns)

    # Save data for inspection in ODS format
    save_data_as_ods(X_train, "Model Data/train_data.ods")
    save_data_as_ods(X_test, "Model Data/test_data.ods")

    # Train the model
    model = build_and_train_model(X_train, y_train, len(features_columns))

    # Predict new bus stops
    new_data = pd.DataFrame({
        "nearest_poi_dist": [0.1, 0.2],
        "temp_normalized": [0.5, 0.6],
        "humidity_normalized": [0.7, 0.8],
        "wind_speed_normalized": [0.3, 0.4],
        "population_density_normalized": [0.9, 0.85],
        "rain": [0.0, 0.1]
    })
    predicted_stops = model.predict(new_data)

    # Visualize the results
    visualize_results_on_map(candidate_stops, predicted_stops, "Template/predicted_bus_stops.html",
                             city_center=[49.8930, 10.9028])
    print("Map saved.")


if __name__ == "__main__":
    main()

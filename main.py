import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import requests


# Load data
def load_data(population_density_file, poi_file, weather_file):
    population_density = pd.read_csv(population_density_file)  # Columns: Latitude, Longitude, Density
    pois = pd.read_csv(poi_file)  # Columns: Latitude, Longitude, POIType, Importance (optional)
    weather = pd.read_csv(weather_file)  # Columns: Latitude, Longitude, WeatherCondition, TimeOfDay, ImpactScore

    return population_density, pois, weather


# Example file paths
population_density_file = "population_density.csv"
poi_file = "pois.csv"
weather_file = "weather_data.csv"

population_density, pois, weather = load_data(population_density_file, poi_file, weather_file)


# Aggregate POI data (if Importance is available, weight POIs by it)
def aggregate_poi_data(pois):
    if 'Importance' in pois.columns:
        pois['WeightedPOI'] = pois['Importance']
    else:
        pois['WeightedPOI'] = 1

    poi_density = pois.groupby(['Latitude', 'Longitude']).sum().reset_index()
    return poi_density


poi_density = aggregate_poi_data(pois)

# Merge population density, POI data, and weather data
combined_data = pd.merge(
    population_density, poi_density,
    on=['Latitude', 'Longitude'], how='outer'
).fillna(0)  # Fill missing values with 0

combined_data = pd.merge(
    combined_data, weather,
    on=['Latitude', 'Longitude'], how='left'
).fillna(0)  # Fill missing weather data with 0

# Calculate a score for potential bus stop locations
combined_data['Score'] = (
        combined_data['Density'] +
        combined_data['WeightedPOI'] +
        combined_data['ImpactScore']
)


# Preprocessing
def preprocess_data(data):
    features = data[['Latitude', 'Longitude', 'Score']].values
    scaler = StandardScaler()
    features[:, 2] = scaler.fit_transform(features[:, 2].reshape(-1, 1)).flatten()
    return features


features = preprocess_data(combined_data)

# Clustering with KMeans to identify optimal bus stop locations
kmeans = KMeans(n_clusters=10, random_state=42)  # Adjust the number of clusters as needed
combined_data['Cluster'] = kmeans.fit_predict(features)

# Determine potential bus stop locations (centroids of clusters)
bus_stop_locations = kmeans.cluster_centers_[:, :2]

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(combined_data['Latitude'], combined_data['Longitude'], c=combined_data['Cluster'], cmap='viridis', s=10,
            label='Data Points')
plt.scatter(bus_stop_locations[:, 0], bus_stop_locations[:, 1], color='red', label='Proposed Bus Stops', marker='x')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Proposed Bus Stop Locations Based on Population Density, POI, and Weather')
plt.legend()
plt.colorbar(label='Cluster')
plt.show()

# Save proposed bus stop locations
proposed_bus_stops = pd.DataFrame(bus_stop_locations, columns=['Latitude', 'Longitude'])
proposed_bus_stops.to_csv("proposed_bus_stops.csv", index=False)

print("Proposed bus stop locations saved to 'proposed_bus_stops.csv'")

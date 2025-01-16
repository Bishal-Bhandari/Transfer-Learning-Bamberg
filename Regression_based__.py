import pandas as pd
import requests
import overpy
import folium
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load population density data from ODS
data = pd.read_excel("Training Data/final_busStop_density.ods")


# Function to get OSM road data using Overpass API
def get_osm_roads(bbox):
    api = overpy.Overpass()
    result = api.query(f"""
        way["highway"!="footpath"]["highway"!="cycleway"](
            {bbox[1]}, {bbox[0]}, {bbox[3]}, {bbox[2]}
        );
        (._;>;);
        out body;
    """)
    return result


# Function to create features for machine learning
def create_features(lat, lon, density, roads):
    """
    Creates features for the machine learning model.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        density: Population density (1-5).
        roads: OSM road data.

    Returns:
        A dictionary of features.
    """
    features = {}
    features['density'] = density

    # Calculate distance to nearest road
    point = Point(lon, lat)
    min_distance = float('inf')
    for way in roads.ways:
        line = LineString([(node.lon, node.lat) for node in way.nodes])
        nearest_point = nearest_points(point, line)[0]
        distance = point.distance(nearest_point)
        min_distance = min(min_distance, distance)
    features['distance_to_road'] = min_distance

    # Add more features as needed (e.g., distance to landmarks, road type)

    return features


# Create features for each location
features_list = []
for index, row in data.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    density = row['Density']

    # Get OSM road data for a bounding box around the location
    bbox = (lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01)
    roads = get_osm_roads(bbox)
    features = create_features(lat, lon, density, roads)
    features_list.append(features)

# Create a DataFrame from the features
features_df = pd.DataFrame(features_list)

# Split data into training and testing sets
X = features_df[['density', 'distance_to_road']]  # Features
y = data['name']  # Target variable (0: no bus stop, 1: bus stop)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict probabilities for all locations
probabilities = model.predict_proba(features_df[['density', 'distance_to_road']])[:, 1]

# Create a Folium map
map_bamberg = folium.Map(location=[49.89, 10.89], zoom_start=13)

# Add markers with probabilities
for index, row in data.iterrows():
    lat = row['lat']
    lon = row['lon']
    probability = probabilities[index]
    folium.CircleMarker(
        location=[lat, lon],
        radius=probability * 5,  # Adjust radius based on probability
        color='blue',
        fill=True,
        fill_opacity=0.5
    ).add_to(map_bamberg)

# Display the map
map_bamberg.save("Template/regression_bus_stop_predictions.html")

# Save results to ODS
results_df = data.copy()
results_df['bus_stop_probability'] = probabilities
results_df.to_excel("Model Data/regression_bus_stop_predictions.ods", index=False)

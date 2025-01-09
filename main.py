import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Load data
def load_data(bus_stop_file, population_density_file, poi_file):
    bus_stops = pd.read_csv(bus_stop_file)  # Columns: Latitude, Longitude, StopName
    population_density = pd.read_csv(population_density_file)  # Columns: Latitude, Longitude, Density
    pois = pd.read_csv(poi_file)  # Columns: Latitude, Longitude, POIType

    return bus_stops, population_density, pois


# Example file paths
bus_stop_file = "bus_stops.csv"
population_density_file = "population_density.csv"
poi_file = "pois.csv"

bus_stops, population_density, pois = load_data(bus_stop_file, population_density_file, poi_file)

# Combine data into a single DataFrame
data = pd.concat([
    bus_stops[['Latitude', 'Longitude']],
    population_density[['Latitude', 'Longitude']],
    pois[['Latitude', 'Longitude']]
], axis=0, ignore_index=True)

data['Type'] = [
    'BusStop' if i < len(bus_stops) else \
        'PopulationDensity' if i < len(bus_stops) + len(population_density) else \
            'POI' for i in range(len(data))
]


# Preprocessing
def preprocess_data(data):
    features = data[['Latitude', 'Longitude']].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features


features = preprocess_data(data)

# Transfer Learning using ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Generate synthetic images from lat/lon as dummy data
# In practice, this would use a meaningful transformation of lat/lon into image-like data.
def generate_synthetic_images(features, num_samples):
    images = np.zeros((num_samples, 224, 224, 3))
    for i in range(num_samples):
        lat, lon = features[i]
        images[i, :, :, 0] = lat  # Red channel
        images[i, :, :, 1] = lon  # Green channel
        images[i, :, :, 2] = lat + lon  # Blue channel
    return images


images = generate_synthetic_images(features, len(features))
images = preprocess_input(images)

# Extract features
feature_maps = base_model.predict(images)
flattened_features = feature_maps.reshape(len(features), -1)

# Clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(flattened_features)
data['Cluster'] = clusters

# Classification example
X_train, X_test, y_train, y_test = train_test_split(flattened_features, clusters, test_size=0.2, random_state=42)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Visualization
plt.scatter(data['Latitude'], data['Longitude'], c=data['Cluster'], cmap='viridis', s=10)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Clustered Data Points')
plt.colorbar(label='Cluster')
plt.show()

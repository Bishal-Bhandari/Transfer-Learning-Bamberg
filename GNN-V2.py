import pandas as pd
import osmnx as ox
import folium
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from scipy.spatial import cKDTree
from tqdm import tqdm
import geopandas as gpd
from joblib import Parallel, delayed
from shapely.geometry import Polygon, MultiPolygon

tqdm.pandas()

def load_grid_data(file_path):
    df = pd.read_excel(file_path, engine='odf',
                       dtype={'Latitude': np.float32, 'Longitude': np.float32, 'Density': np.int32})
    scaler = MinMaxScaler()
    df['Normalized_Density'] = scaler.fit_transform(df[['Density']]).astype(np.float32)
    return df

def get_osm_data(place_name="Bamberg, Germany"):
    graph = ox.graph_from_place(place_name, network_type='drive', simplify=True)
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    poi_tags = {'amenity': True, 'shop': True, 'public_transport': True, 'tourism': True}
    pois = ox.features_from_place(place_name, poi_tags)
    pois = pois[pois.geometry.notnull()].copy()
    pois['geometry'] = pois.geometry.apply(lambda g: g.centroid if isinstance(g, (Polygon, MultiPolygon)) else g)
    pois = pois.explode(index_parts=True).reset_index(drop=True)
    return graph, nodes, pois

def calculate_poi_density(candidates, pois, radius=500):
    if pois.empty:
        candidates['POI_Count'] = 0
        return candidates
    tree = cKDTree(pois[['geometry'].apply(lambda g: (g.x, g.y))].tolist())
    counts = tree.query_ball_point(candidates[['Longitude', 'Latitude']].values, r=radius, return_length=True)
    candidates['POI_Count'] = counts
    return candidates

def generate_candidates(graph, grid_data):
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    grid_data['node_id'] = ox.distance.nearest_nodes(graph, grid_data.Longitude, grid_data.Latitude)
    candidate_nodes = nodes[~nodes.index.isin(grid_data['node_id'])].sample(frac=0.3).reset_index()
    candidates = candidate_nodes[['y', 'x']]
    candidates.columns = ['Latitude', 'Longitude']
    return candidates

def prepare_graph_data(graph, combined):
    unique_nodes = combined['node_id'].unique().tolist()
    node_id_to_idx = {nid: idx for idx, nid in enumerate(unique_nodes)}
    edge_list = [[node_id_to_idx[u], node_id_to_idx[v]] for u, v in graph.edges() if u in node_id_to_idx and v in node_id_to_idx]
    node_features = combined.groupby('node_id').agg({'Normalized_Density': 'mean', 'POI_Count': 'mean'}).reindex(unique_nodes).fillna(0)
    node_features = MinMaxScaler().fit_transform(node_features)
    return Data(x=torch.tensor(node_features, dtype=torch.float32), edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous()), combined, node_id_to_idx

class BusStopPredictor(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim//2)
        self.predictor = torch.nn.Sequential(torch.nn.Linear(hidden_dim//2, 32), torch.nn.ReLU(), torch.nn.Linear(32, output_dim))
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        return torch.sigmoid(self.predictor(x))

def train_model(data, epochs=500, patience=100):
    model = BusStopPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    target_weights = data.x[:, 0] * 0.6 + data.x[:, 1] * 0.4
    best_loss = float('inf')
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(data).squeeze()
        loss = torch.mean((pred - target_weights) ** 2)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1} Loss {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    return model

def predict_and_adjust(model, data, graph, candidates, node_id_to_idx):
    model.eval()
    with torch.no_grad():
        predictions = model(data).squeeze().numpy()
    candidates['pred_prob'] = candidates['node_id'].map(lambda nid: predictions[node_id_to_idx[nid]] if nid in node_id_to_idx else 0.0)
    final_stops = candidates[candidates['pred_prob'] > 0.5]
    if final_stops.empty:
        final_stops = candidates.nlargest(1, 'pred_prob')
    return final_stops

def main():
    grid_data = load_grid_data("grid_density_data.ods")
    graph, _, pois = get_osm_data()
    candidates = generate_candidates(graph, grid_data)
    candidates = calculate_poi_density(candidates, pois)
    combined = pd.concat([grid_data, candidates], ignore_index=True)
    graph_data, combined, node_id_to_idx = prepare_graph_data(graph, combined)
    model = train_model(graph_data)
    predictions = predict_and_adjust(model, graph_data, graph, combined, node_id_to_idx)
    m = folium.Map(location=[49.8988, 10.9028], zoom_start=14)
    for _, stop in predictions.iterrows():
        folium.CircleMarker(location=[stop.Latitude, stop.Longitude], radius=6, color='#ff0000' if stop.pred_prob > 0.7 else '#ffa500', fill=True, opacity=0.7, popup=f"Score: {stop.pred_prob:.2f}").add_to(m)
    m.save("Template/optimized_predictions.html")

if __name__ == "__main__":
    main()

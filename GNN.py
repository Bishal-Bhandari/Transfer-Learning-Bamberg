import pandas as pd
import osmnx as ox
import folium
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm
import geopandas as gpd
from joblib import Parallel, delayed
from shapely.geometry import Polygon, Point, MultiPolygon
import pyexcel as pe

ox.settings.log_console = True
ox.settings.use_cache = True
tqdm.pandas()


# 1. Load Grid Data
def load_grid_data(grid_file):
    grid_df = pd.read_excel(grid_file, engine='odf',
                            dtype={'min_lat': np.float32, 'max_lat': np.float32,
                                   'min_lon': np.float32, 'max_lon': np.float32,
                                   'density_rank': np.int8})

    # Filter out rows with NaN values in critical columns
    grid_df = grid_df.dropna(subset=['min_lat', 'max_lat', 'min_lon', 'max_lon', 'density_rank'])

    # Create polygons
    grid_df['geometry'] = grid_df.apply(lambda row: Polygon([
        (row['min_lon'], row['min_lat']),
        (row['max_lon'], row['min_lat']),
        (row['max_lon'], row['max_lat']),
        (row['min_lon'], row['max_lat'])
    ]), axis=1)

    return gpd.GeoDataFrame(grid_df, geometry='geometry', crs="EPSG:4326")


# 2. Load Bus Stops
def load_bus_stops(stop_file, grid_gdf, graph):
    stops = pd.read_excel(stop_file, engine='odf')

    # Column mapping for flexible reading
    col_map = {'Latitude': ['Latitude', 'lat', 'stop_lat'],
               'Longitude': ['Longitude', 'lon', 'stop_lon', 'lng']}

    lat_col = next((c for c in col_map['Latitude'] if c in stops.columns), None)
    lon_col = next((c for c in col_map['Longitude'] if c in stops.columns), None)

    if not lat_col or not lon_col:
        raise ValueError(f"Latitude/Longitude columns not found. Available columns: {stops.columns.tolist()}")

    stops = stops.rename(columns={lat_col: 'Latitude', lon_col: 'Longitude'})

    # Filter out rows with NaN in 'Latitude' or 'Longitude'
    stops = stops.dropna(subset=['Latitude', 'Longitude'])

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops.Longitude, stops.Latitude), crs="EPSG:4326")

    # Join with grid
    joined = gpd.sjoin(gdf, grid_gdf[['geometry', 'density_rank']], how='left', predicate='within')

    # Normalize density
    joined['density_rank'] = joined['density_rank'].fillna(1).astype(np.int8)
    joined['Normalized_Density'] = MinMaxScaler().fit_transform(joined[['density_rank']])

    # Assign nearest node
    if joined[['Longitude', 'Latitude']].isna().any().any():
        raise ValueError("Missing Longitude/Latitude values in bus stops data.")

    joined['node_id'] = ox.distance.nearest_nodes(graph, joined['Longitude'], joined['Latitude'])

    return joined.drop(columns='index_right')


# 3. Generate Candidates
def generate_candidates(graph, existing_stops, grid_gdf):
    """Generate candidates based on grid density"""
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)

    # Filter out existing stop nodes
    candidate_nodes = nodes[~nodes.index.isin(existing_stops['node_id'])]

    # Create GeoDataFrame with geometry
    candidate_gdf = gpd.GeoDataFrame(
        candidate_nodes[['y', 'x']].reset_index(),
        geometry=gpd.points_from_xy(candidate_nodes.x, candidate_nodes.y),
        crs="EPSG:4326"
    )

    # Join with grid data
    candidates = gpd.sjoin(candidate_gdf, grid_gdf[['geometry', 'density_rank']],
                           how='left', predicate='within')

    # Ensure missing `density_rank` values are handled
    candidates['density_rank'] = candidates['density_rank'].fillna(1).astype(int)

    sample_weights = {5: 0.8, 4: 0.6, 3: 0.4, 2: 0.2, 1: 0.1}
    sampled = [candidates[candidates['density_rank'] == rank].sample(frac=weight, replace=True)
               for rank, weight in sample_weights.items() if not candidates[candidates['density_rank'] == rank].empty]

    final_candidates = pd.concat(sampled)
    final_candidates['Normalized_Density'] = final_candidates['density_rank'] / 5.0

    # Ensure `density_rank` remains in the final DataFrame
    return final_candidates[['y', 'x', 'Normalized_Density', 'density_rank']].rename(
        columns={'y': 'Latitude', 'x': 'Longitude'}
    )


# 4. Calculate POI Density
def calculate_poi_density(points_gdf, pois_gdf, radius=500):
    """Calculate POI density around candidate points."""

    if pois_gdf.empty:
        points_gdf['POI_Count'] = 0
        return points_gdf

    # Ensure points_gdf is a GeoDataFrame
    if not isinstance(points_gdf, gpd.GeoDataFrame):
        points_gdf = gpd.GeoDataFrame(
            points_gdf,
            geometry=gpd.points_from_xy(points_gdf.Longitude, points_gdf.Latitude),
            crs="EPSG:4326"
        )

    # Convert polygons to centroids for POIs
    pois_gdf['geometry'] = pois_gdf['geometry'].apply(
        lambda geom: geom.centroid if geom.geom_type in ['Polygon', 'MultiPolygon'] else geom
    )

    # Filter valid POI points
    pois_gdf = pois_gdf[pois_gdf.geometry.notnull() & (pois_gdf.geometry.geom_type == 'Point')]

    if pois_gdf.empty:
        points_gdf['POI_Count'] = 0
        return points_gdf

    # Convert to UTM for accurate distance calculations
    utm_crs = points_gdf.estimate_utm_crs()
    points_utm = points_gdf.to_crs(utm_crs)
    pois_utm = pois_gdf.to_crs(utm_crs)

    # Create spatial index
    poi_coords = np.array([(geom.x, geom.y) for geom in pois_utm.geometry])
    tree = cKDTree(poi_coords)

    def count_pois(row):
        return len(tree.query_ball_point([row.geometry.x, row.geometry.y], r=radius))

    points_utm['POI_Count'] = Parallel(n_jobs=-1)(
        delayed(count_pois)(row) for _, row in tqdm(points_utm.iterrows(), total=len(points_utm))
    )

    return points_gdf.merge(points_utm[['POI_Count']], left_index=True, right_index=True)


# 5. Prepare Graph Data
def prepare_graph_data(graph, combined_data):
    """Create graph structure for GNN ensuring valid edges."""
    combined_data['node_id'] = ox.distance.nearest_nodes(
        graph,
        combined_data['Longitude'],
        combined_data['Latitude']
    )

    # Filter only nodes that exist in the dataset
    valid_nodes = set(combined_data['node_id'])
    edge_list = [[u, v] for u, v in graph.edges() if u in valid_nodes and v in valid_nodes]

    unique_nodes = sorted(valid_nodes)
    node_id_to_idx = {nid: idx for idx, nid in enumerate(unique_nodes)}

    # Ensure edge indices are mapped correctly
    edge_list = [[node_id_to_idx[u], node_id_to_idx[v]] for u, v in edge_list if u in node_id_to_idx and v in node_id_to_idx]

    features = combined_data.groupby('node_id')[['Normalized_Density', 'POI_Count']].mean()
    features = features.reindex(unique_nodes).fillna(0).values  # Ensure correct ordering

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    )

    return data, unique_nodes


# 6. Train Model
class BusStopPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 128)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.conv2 = GCNConv(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.predictor = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout2(x)
        return self.predictor(x).squeeze()  # Return raw logits


# 6. Training & Prediction
def train_model(model, data, epochs=300):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    # Calculate class weights and move to correct device
    pos_weight = torch.tensor([(len(data.y) - data.y.sum()) / data.y.sum()]).to(data.x.device)

    # Use BCEWithLogitsLoss instead of BCELoss
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)  # Now returns logits (without sigmoid)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}: Loss {loss.item():.4f}')

    model.load_state_dict(torch.load('best_model.pth'))
    return model


# 7. Visualization Function
def create_results_map(existing_stops, predictions, grid_gdf):
    existing_stops = existing_stops.dropna(subset=['Latitude', 'Longitude'])
    predictions = predictions.dropna(subset=['Latitude', 'Longitude'])

    existing_stops[['Latitude', 'Longitude']] = existing_stops[['Latitude', 'Longitude']].apply(pd.to_numeric,
                                                                                                errors='coerce')
    predictions[['Latitude', 'Longitude']] = predictions[['Latitude', 'Longitude']].apply(pd.to_numeric,
                                                                                          errors='coerce')

    existing_stops = existing_stops.dropna(subset=['Latitude', 'Longitude'])
    predictions = predictions.dropna(subset=['Latitude', 'Longitude'])

    avg_lat = np.mean([existing_stops.Latitude.mean(), predictions.Latitude.mean()])
    avg_lon = np.mean([existing_stops.Longitude.mean(), predictions.Longitude.mean()])

    if np.isnan(avg_lat) or np.isnan(avg_lon):
        avg_lat, avg_lon = 50.8503, 4.3517

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14, tiles='CartoDB positron')

    grid_layer = folium.FeatureGroup(name='Density Grid')
    for _, grid in grid_gdf.iterrows():
        if all(col in grid for col in ['min_lat', 'min_lon', 'max_lat', 'max_lon', 'density_rank']):
            grid_layer.add_child(
                folium.Rectangle(
                    bounds=[[grid['min_lat'], grid['min_lon']],
                            [grid['max_lat'], grid['max_lon']]],
                    color='#ff0000',
                    fill=True,
                    fill_color='YlOrRd',
                    fill_opacity=0.2 * grid['density_rank'],
                    popup=f"Density Rank: {grid['density_rank']}"
                )
            )
    m.add_child(grid_layer)

    existing_layer = folium.FeatureGroup(name='Existing Stops')
    for _, stop in existing_stops.iterrows():
        existing_layer.add_child(
            folium.CircleMarker(
                location=[stop['Latitude'], stop['Longitude']],
                radius=6,
                color='#00cc00',
                fill=True,
                popup=f"Name: {stop.get('Stop_Name', 'N/A')}<br>"
                      f"Density Rank: {stop['density_rank']}<br>"
                      f"POI Count: {stop['POI_Count']}"
            )
        )
    m.add_child(existing_layer)

    pred_layer = folium.FeatureGroup(name='Predicted Stops')
    for _, pred in predictions.iterrows():
        pred_layer.add_child(
            folium.CircleMarker(
                location=[pred['Latitude'], pred['Longitude']],
                radius=5 + 8 * pred['pred_prob'],
                color='#0066cc',
                fill=True,
                fill_opacity=0.7,
                popup=f"Probability: {pred['pred_prob']:.2f}<br>"
                      f"Density Rank: {pred['density_rank']}<br>"
                      f"POIs: {pred['POI_Count']}"
            )
        )
    m.add_child(pred_layer)

    folium.LayerControl(collapsed=False).add_to(m)
    title_html = '''
         <h3 align="center" style="font-size:16px"><b>Bus Stop Predictions</b></h3>
         <div style="text-align:center;">
             <span style="color: #00cc00;">■</span> Existing Stops &nbsp;
             <span style="color: #0066cc;">■</span> Predicted Stops
         </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    return m


#8. Save result
def save_predictions(predictions, output_file):
    """Save predictions to ODS format"""
    predictions = predictions.astype({
        'Latitude': np.float32,
        'Longitude': np.float32,
        'density_rank': np.int8,
        'POI_Count': np.int16,
        'pred_prob': np.float32
    })

    # Convert pandas DataFrame to a dictionary for pyexcel to handle
    predictions_dict = predictions.to_dict(orient='records')

    # Save to ODS using pyexcel
    pe.save_as(records=predictions_dict, dest_file_name=output_file)


# 9. Main Workflow
def main():
    grid_gdf = load_grid_data("Training Data/city_grid_density.ods")
    graph = ox.graph_from_place("Brussels, Belgium", network_type='drive', simplify=True)
    bus_stops = load_bus_stops("Training Data/stib_stops.ods", grid_gdf, graph)
    pois = ox.features_from_place("Brussels, Belgium", tags={'amenity': True})

    bus_stops = calculate_poi_density(bus_stops, pois)
    candidates = generate_candidates(graph, bus_stops, grid_gdf)
    candidates = calculate_poi_density(candidates, pois)

    # Scale POI_Count across all data
    poi_scaler = MinMaxScaler()
    combined = pd.concat([
        bus_stops[['Latitude', 'Longitude', 'Normalized_Density', 'POI_Count', 'density_rank']],
        candidates[['Latitude', 'Longitude', 'Normalized_Density', 'POI_Count', 'density_rank']]
    ], ignore_index=True)
    combined['POI_Count'] = poi_scaler.fit_transform(combined[['POI_Count']])

    graph_data, unique_nodes = prepare_graph_data(graph, combined)

    # Assign correct labels (1 for existing stops, 0 for candidates)
    existing_node_ids = set(bus_stops['node_id'])
    graph_data.y = torch.tensor(
        [1 if node_id in existing_node_ids else 0 for node_id in unique_nodes],
        dtype=torch.float32
    )

    model = BusStopPredictor()
    model = train_model(model, graph_data)

    with torch.no_grad():
        logits = model(graph_data).numpy()
        pred_prob = torch.sigmoid(torch.tensor(logits)).numpy()  # Apply sigmoid here

    predictions_df = pd.DataFrame({
        'node_id': unique_nodes,
        'pred_prob': pred_prob
    }).merge(
        combined[['node_id', 'Latitude', 'Longitude', 'density_rank', 'POI_Count']],
        on='node_id',
        how='left'
    )

    # Filter out existing stops and select high-probability candidates
    new_predictions = predictions_df[
        (~predictions_df['node_id'].isin(existing_node_ids)) &
        (predictions_df['pred_prob'] > 0.5)
    ]

    save_predictions(new_predictions, 'Model Data/bus_stop_predictions.ods')

    result_map = create_results_map(
        bus_stops,
        new_predictions[['Latitude', 'Longitude', 'density_rank', 'POI_Count', 'pred_prob']],
        grid_gdf
    )
    result_map.save("Template/bus_stop_predictions_map.html")


if __name__ == "__main__":
    main()



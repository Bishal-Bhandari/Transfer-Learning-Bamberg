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
from shapely.geometry import Polygon, Point, MultiPolygon

# Install required packages if missing:
# pip install odfpy geopandas osmnx torch_geometric folium joblib

ox.settings.log_console = True
ox.settings.use_cache = True
tqdm.pandas()


# 1. ODS Data Loaders ---------------------------------------------------------
def load_grid_data(grid_file):
    """Load grid data from ODS file with density ranks"""
    grid_df = pd.read_excel(grid_file, engine='odf',
                            dtype={'min_lat': np.float32,
                                   'max_lat': np.float32,
                                   'min_lon': np.float32,
                                   'max_lon': np.float32,
                                   'density_rank': np.int8})

    # Create grid polygons
    grid_geoms = []
    for _, row in grid_df.iterrows():
        poly = Polygon([
            (row['min_lon'], row['min_lat']),
            (row['max_lon'], row['min_lat']),
            (row['max_lon'], row['max_lat']),
            (row['min_lon'], row['max_lat'])
        ])
        grid_geoms.append(poly)

    return gpd.GeoDataFrame(
        grid_df,
        geometry=grid_geoms,
        crs="EPSG:4326"
    )


def load_bus_stops(stop_file, grid_gdf):
    """Load existing stops from ODS and add grid density"""
    stops = pd.read_excel(stop_file, engine='odf',
                          dtype={'Latitude': np.float32,
                                 'Longitude': np.float32})

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.Longitude, stops.Latitude),
        crs="EPSG:4326"
    )

    # Spatial join with grid data
    joined = gpd.sjoin(gdf, grid_gdf[['geometry', 'density_rank']],
                       how='left', predicate='within')

    # Fill missing ranks and normalize
    joined['density_rank'] = joined['density_rank'].fillna(1).astype(np.int8)
    scaler = MinMaxScaler()
    joined['Normalized_Density'] = scaler.fit_transform(joined[['density_rank']])

    return joined.drop(columns='index_right')


# 2. Candidate Generation -----------------------------------------------------
def generate_candidates(graph, existing_stops, grid_gdf):
    """Generate candidates based on grid density"""
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)

    # Filter out existing stop nodes
    candidate_nodes = nodes[~nodes.index.isin(existing_stops['node_id'])]
    candidate_gdf = gpd.GeoDataFrame(
        candidate_nodes[['y', 'x']].reset_index(),
        geometry=gpd.points_from_xy(candidate_nodes.x, candidate_nodes.y),
        crs="EPSG:4326"
    )

    # Join with grid data
    candidates = gpd.sjoin(candidate_gdf, grid_gdf[['geometry', 'density_rank']],
                           how='left', predicate='within')

    # Density-based sampling weights
    sample_weights = {
        5: 0.8,  # Highest density
        4: 0.6,
        3: 0.4,
        2: 0.2,
        1: 0.1  # Lowest density
    }

    sampled = []
    for rank, weight in sample_weights.items():
        subset = candidates[candidates['density_rank'] == rank]
        if not subset.empty:
            sampled.append(subset.sample(frac=weight))

    final_candidates = pd.concat(sampled)

    # Add normalized density
    final_candidates['Normalized_Density'] = final_candidates['density_rank'] / 5.0
    return final_candidates[['y', 'x', 'Normalized_Density']].rename(
        columns={'y': 'Latitude', 'x': 'Longitude'})


# 3. POI Density Calculation --------------------------------------------------
def calculate_poi_density(points_gdf, pois_gdf, radius=500):
    """Calculate POI density in parallel"""
    if pois_gdf.empty:
        points_gdf['POI_Count'] = 0
        return points_gdf

    utm_crs = points_gdf.estimate_utm_crs()
    points_utm = points_gdf.to_crs(utm_crs)
    pois_utm = pois_gdf.to_crs(utm_crs)

    poi_coords = np.array([(geom.x, geom.y) for geom in pois_utm.geometry])
    tree = cKDTree(poi_coords)

    def count_pois(row):
        point = row.geometry
        return len(tree.query_ball_point([point.x, point.y], r=radius))

    points_utm['POI_Count'] = Parallel(n_jobs=-1)(
        delayed(count_pois)(row) for _, row in tqdm(points_utm.iterrows())
    )

    return points_gdf.merge(points_utm['POI_Count'], left_index=True, right_index=True)


# 4. Graph Data Preparation ---------------------------------------------------
def prepare_graph_data(graph, combined_data):
    """Create graph structure for GNN"""
    combined_data['node_id'] = ox.distance.nearest_nodes(
        graph,
        combined_data['Longitude'],
        combined_data['Latitude']
    )

    node_features = combined_data.groupby('node_id').agg({
        'Normalized_Density': 'max',
        'POI_Count': 'max'
    }).reset_index()

    edge_list = []
    for u, v, _ in graph.edges(data=True):
        if u in node_features['node_id'].values and v in node_features['node_id'].values:
            edge_list.append([u, v])

    unique_nodes = node_features['node_id'].unique()
    node_id_to_idx = {nid: idx for idx, nid in enumerate(unique_nodes)}

    features = node_features[['Normalized_Density', 'POI_Count']].values
    features = np.nan_to_num(features, nan=0)

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
        node_id_mapping=node_id_to_idx
    ), node_features


# 5. GNN Model ----------------------------------------------------------------
class BusStopPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 128)
        self.conv2 = GCNConv(128, 64)
        self.attention = torch.nn.Linear(64, 1)
        self.predictor = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        attn_weights = torch.sigmoid(self.attention(x))
        x = x * attn_weights
        return torch.sigmoid(self.predictor(x)).squeeze()


# 6. Training & Prediction ----------------------------------------------------
def train_model(model, data, epochs=300):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}: Loss {loss.item():.4f}')

    model.load_state_dict(torch.load('best_model.pth'))
    return model


# 7. Visualization Function ---------------------------------------------------
def create_results_map(existing_stops, predictions, grid_gdf):
    """Create interactive Folium map with layers"""
    # Calculate map center
    avg_lat = np.mean([existing_stops.Latitude.mean(), predictions.Latitude.mean()])
    avg_lon = np.mean([existing_stops.Longitude.mean(), predictions.Longitude.mean()])

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14, tiles='CartoDB positron')

    # Add grid layer
    grid_layer = folium.FeatureGroup(name='Density Grid')
    for _, grid in grid_gdf.iterrows():
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

    # Add existing stops layer
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

    # Add predictions layer
    pred_layer = folium.FeatureGroup(name='Predicted Stops')
    for _, pred in predictions.iterrows():
        pred_layer.add_child(
            folium.CircleMarker(
                location=[pred['Latitude'], pred['Longitude']],
                radius=5 + 8 * pred['pred_prob'],  # Scale radius by probability
                color='#0066cc',
                fill=True,
                fill_opacity=0.7,
                popup=f"Probability: {pred['pred_prob']:.2f}<br>"
                      f"Density Rank: {pred['density_rank']}<br>"
                      f"POIs: {pred['POI_Count']}"
            )
        )
    m.add_child(pred_layer)

    # Add layer control and title
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

#8. Save result---------------------------------------------------------------
def save_predictions(predictions, output_file):
    """Save predictions to ODS with optimized formatting"""
    predictions = predictions.astype({
        'Latitude': np.float32,
        'Longitude': np.float32,
        'density_rank': np.int8,
        'POI_Count': np.int16,
        'pred_prob': np.float32
    })

    with pd.ExcelWriter(output_file, engine='odf') as writer:
        predictions.to_excel(writer, index=False, sheet_name='Predictions',
                             float_format="%.3f")

        # Formatting
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        worksheet.column(0, 0).width = 12  # Latitude
        worksheet.column(1, 1).width = 12  # Longitude
        worksheet.column(2, 2).width = 10  # Density Rank
        worksheet.column(3, 3).width = 10  # POI Count
        worksheet.column(4, 4).width = 14  # Probability


# 9. Updated Main Workflow ----------------------------------------------------
def main():
    # Load input data
    grid_gdf = load_grid_data("Training Data/city_grid_density.ods")
    bus_stops = load_bus_stops("Training Data/existing_stops.ods", grid_gdf)

    # Get OSM data
    graph = ox.graph_from_place("Bamberg, Germany", network_type='drive', simplify=True)
    pois = ox.features_from_place("Bamberg, Germany", tags={'amenity': True})

    # Generate candidates
    candidates = generate_candidates(graph, bus_stops, grid_gdf)
    candidates = calculate_poi_density(candidates, pois)

    # Combine datasets
    combined = pd.concat([
        bus_stops[['Latitude', 'Longitude', 'Normalized_Density', 'POI_Count', 'density_rank']],
        candidates[['Latitude', 'Longitude', 'Normalized_Density', 'POI_Count', 'density_rank']]
    ], ignore_index=True)

    # Prepare graph data
    graph_data, node_features = prepare_graph_data(graph, combined)

    # Create labels
    graph_data.y = torch.tensor(
        [1 if id in bus_stops['node_id'].values else 0 for id in node_features['node_id']],
        dtype=torch.float32
    )

    # Train model
    model = BusStopPredictor()
    model = train_model(model, graph_data)

    # Generate predictions
    with torch.no_grad():
        node_features['pred_prob'] = model(graph_data).numpy()

    # Filter and save results
    predictions = node_features[node_features['pred_prob'] > 0.5]
    save_predictions(predictions[['Latitude', 'Longitude', 'density_rank',
                                  'POI_Count', 'pred_prob']],
                     "bus_stop_predictions.ods")

    # Create and save interactive map
    result_map = create_results_map(
        bus_stops,
        predictions[['Latitude', 'Longitude', 'density_rank', 'POI_Count', 'pred_prob']],
        grid_gdf
    )
    result_map.save("bus_stop_predictions_map.html")


if __name__ == "__main__":
    main()
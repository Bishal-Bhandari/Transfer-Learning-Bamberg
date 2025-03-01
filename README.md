## Transfer Learning for transportation

## Introduction

This project implements an automated system to predict optimal bus stop locations in urban areas. By integrating multiple data sources—including road network information from OpenStreetMap, weather forecasts from OpenWeatherMap, city grid density, and points-of-interest (POI) data—the system uses a pretrained Graph Neural Network (GNN) model to identify candidate bus stops. The predictions are visualized through interactive maps, providing urban planners and transport authorities with a data-driven tool to enhance public transit planning.

## Methodology

1. **Data Collection:**  
   - **Road Network:** Downloaded using OSMnx, capturing detailed road and junction data.  
   - **City Grid & Bus Stops:** Loaded from local files to obtain grid density metrics and existing bus stop data.  
   - **POI Data:** Parsed from JSON files to include local attractions and facilities.  
   - **Weather Forecasts:** Retrieved via the OpenWeatherMap API to factor in environmental conditions at the specified time.

2. **Data Processing:**  
   - **Graph Preparation:** The road network graph is cleaned, with node attributes normalized (e.g., geographic coordinates and road type encodings).  
   - **Feature Extraction:** Grid features are computed by combining density rankings, POI influence, and weather data, then normalized for model input.
   - **Bounding Box Expansion:** The city’s geographic bounds are expanded (using a configurable radius) to ensure comprehensive coverage.

3. **Prediction Pipeline:**  
   - **GNN Model:** A GraphSAGE-based model (integrated with batch normalization and dropout layers) predicts bus stop suitability for each road network node.  
   - **Candidate Filtering:** Predictions are refined based on grid density, time-dependent factors, and spatial constraints (e.g., minimum distance thresholds between stops).

4. **Visualization:**  
   - **Interactive Mapping:** Predicted bus stops and city grid data are overlaid on an interactive Folium map.  
   - **Static Output:** Optionally, static maps can be generated using Matplotlib for quick visual reference.
   - **Output Files:** Final predictions are saved in CSV format for further analysis or integration.

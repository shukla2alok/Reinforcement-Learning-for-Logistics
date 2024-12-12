# Rider Delivery Route Visualization with RL and OSRM

## Overview
This project demonstrates a Rider Delivery Route Visualization application. It integrates various APIs and methods to calculate routes, simulate Reinforcement Learning (RL)-optimized paths, and visualize these routes on an interactive map. The application is built using Python and Folium for visualization.

---

## Features
1. **Restaurant, Delivery, and Rider Location Markers**:
   - Displays restaurant, delivery, and rider locations with distinct icons and popups.
   - Includes enhanced tooltips and popups for better user interaction.

2. **Route Visualization**:
   - Simulates RL-optimized routes.
   - Fetches route data from **OSRM API** for real-time routing.
   - Uses **OpenRouteService (ORS)** as an alternative for route calculations.
   
3. **Color-coded Paths**:
   - **Green**: RL-optimized routes.
   - **Blue**: Rider to restaurant paths.
   - **Red**: Restaurant to delivery paths.

4. **Interactive Map**:
   - Generated using Folium.
   - Features a MarkerCluster for grouping markers.
   - Includes a legend explaining map elements.

5. **HTML Export**:
   - Saves the interactive map as an HTML file.

---

## Requirements
To run this project, ensure the following dependencies are installed:

- Python 3.8+
- `folium`
- `requests`
- `random`
- `openrouteservice`

Install dependencies using:
```bash
pip install folium requests openrouteservice
```

---

## Files

### 1. **`rider_delivery_routes_map.html`**
   - Map generated using OSRM API showing RL-optimized routes and standard routes.

### 2. **`rider_delivery_routes_with_rl_map.html`**
   - Map with routes specifically simulating RL-optimized paths and OSRM-based routing.

### 3. **`rider_delivery_routes_map_with_ors.html`**
   - Map generated using OpenRouteService API as an alternative to OSRM.

---

## Code Explanation

### 1. **Data Initialization**:
   - **Locations**: Define sample coordinates for restaurants, riders, and delivery points.
   - Coordinates are simulated for demonstration but can be replaced with real data.

### 2. **Route Fetching**:
#### **OSRM API**
   - Fetches routes using `http://router.project-osrm.org/route/v1/driving`.
   - Utilizes `geometry` data to draw paths.

#### **OpenRouteService API**
   - Fetches routes using the OpenRouteService Python client.
   - Decodes GeoJSON responses into usable coordinate lists.

### 3. **Map Creation**:
   - Initializes a Folium map centered at a sample location (Lucknow).
   - Adds markers for restaurants, deliveries, and riders.
   - Draws routes using Folium `PolyLine` with appropriate colors and tooltips.

### 4. **Simulating RL Routes**:
   - Simulates RL-optimized paths by highlighting specific routes in green.
   - These are overlaid on standard OSRM/ORS routes for comparison.

### 5. **Legend**:
   - Provides a quick visual reference for map elements.
   - Positioned on the bottom-left corner of the map.

---

## How to Run
1. **Setup API Keys**:
   - If using OpenRouteService, set your API key in the `ORS_API_KEY` variable.

2. **Run the Script**:
   - Execute the Python script in your preferred environment:
     ```bash
     python rider_delivery_visualization.py
     ```

3. **View the Map**:
   - Open the generated HTML file(s) in your browser.

---

## Customization
- **Input Data**: Replace `restaurant_locations`, `delivery_locations`, and `rider_locations` with your actual data.
- **Routing Profile**: Adjust the routing profile (e.g., `driving-car`, `cycling`) for OpenRouteService.
- **Styling**: Modify colors, marker icons, and popup content in the Folium map.

---

## Limitations
1. **Simulated RL Routes**: RL-based paths are currently simulated and do not utilize a trained RL model.
2. **Static Data**: The project uses predefined sample coordinates. Real-time data integration is required for deployment.
3. **Dependency on APIs**:
   - Internet connection is required for OSRM and ORS API calls.

---

## Future Improvements
1. **Reinforcement Learning Integration**:
   - Train an RL model to dynamically assign routes based on rider, restaurant, and delivery locations.

2. **Real-time Updates**:
   - Fetch live data for rider, restaurant, and delivery locations.

3. **Advanced Visualizations**:
   - Incorporate heatmaps and dynamic route adjustments.

---

## Acknowledgments
- **Folium**: For creating interactive maps.
- **OSRM API**: For routing and distance calculations.
- **OpenRouteService**: For alternative route planning.

---




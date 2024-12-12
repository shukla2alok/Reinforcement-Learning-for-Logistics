import pandas as pd
import numpy as np

# Random seed for reproducibility
np.random.seed(42)

# Generating synthetic order data
n_orders = 100
orders = pd.DataFrame({
    'order_id': range(1, n_orders + 1),
    'pickup_lat': np.random.uniform(26.80, 26.85, n_orders),  # Random latitudes within Lucknow
    'pickup_lng': np.random.uniform(80.90, 81.00, n_orders),  # Random longitudes within Lucknow
    'dropoff_lat': np.random.uniform(26.80, 26.85, n_orders),
    'dropoff_lng': np.random.uniform(80.90, 81.00, n_orders),
    'order_time': pd.date_range("2024-12-10 08:00:00", periods=n_orders, freq="5min")
})

# Generating synthetic rider data
n_riders = 10
riders = pd.DataFrame({
    'rider_id': range(1, n_riders + 1),
    'current_lat': np.random.uniform(26.80, 26.85, n_riders),
    'current_lng': np.random.uniform(80.90, 81.00, n_riders),
    'availability': np.random.choice([True, False], n_riders, p=[0.8, 0.2])
})

print("Synthetic Orders Data:\n", orders.head())
print("\nSynthetic Riders Data:\n", riders.head())

import folium

# Center point for visualization (Lucknow)
map_center = [26.85, 80.95]
m = folium.Map(location=map_center, zoom_start=12)

# Add pickup and dropoff points
for _, row in orders.iterrows():
    folium.Marker([row['pickup_lat'], row['pickup_lng']], icon=folium.Icon(color='blue', icon='cloud')).add_to(m)
    folium.Marker([row['dropoff_lat'], row['dropoff_lng']], icon=folium.Icon(color='green', icon='ok')).add_to(m)

# Save the map as an HTML file
m.save("delivery_map.html")
print("Map saved as 'delivery_map.html'")

# Normalize latitudes and longitudes
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
orders[['pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng']] = scaler.fit_transform(
    orders[['pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng']]
)

print("Preprocessed Orders Data:\n", orders.head())



# rider assignment
from geopy.distance import geodesic


# Function to assign riders to orders
def assign_riders(orders, riders):
    assignments = []
    for _, order in orders.iterrows():
        min_dist = float('inf')
        assigned_rider = None
        for _, rider in riders[riders['availability']].iterrows():
            distance = geodesic(
                (order['pickup_lat'], order['pickup_lng']),
                (rider['current_lat'], rider['current_lng'])
            ).km
            if distance < min_dist:
                min_dist = distance
                assigned_rider = rider['rider_id']

        if assigned_rider:
            assignments.append((order['order_id'], assigned_rider))
            riders.loc[riders['rider_id'] == assigned_rider, 'availability'] = False

    return pd.DataFrame(assignments, columns=['order_id', 'rider_id'])


assignments = assign_riders(orders, riders)
print("Rider Assignments:\n", assignments)


#DQN------------

# Define states, actions, and rewards
states = orders[['pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng']].values
actions = list(range(n_riders))  # Actions correspond to rider IDs
rewards = np.random.uniform(-1, 1, size=(len(states), len(actions)))  # Example reward structure

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network for Q-Learning
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize Q-network and optimizer
state_size = states.shape[1]
action_size = len(actions)
q_network = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop (simplified)
for epoch in range(100):
    state_tensor = torch.FloatTensor(states)
    q_values = q_network(state_tensor)
    target_q_values = q_values.clone().detach()
    loss = criterion(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Q-network training complete.")


#visulization of result
import matplotlib.pyplot as plt

# Visualize delivery times


import folium
import random

# Define Lucknow's geographical boundaries
latitude_range = (26.50, 27.17)
longitude_range = (80.50, 81.22)

# Generate random restaurant and delivery points
n_riders = 10  # Number of riders
n_orders = 30  # Number of orders
restaurant_locations = [
    (random.uniform(*latitude_range), random.uniform(*longitude_range)) for _ in range(5)
]
delivery_locations = [
    (random.uniform(*latitude_range), random.uniform(*longitude_range)) for _ in range(n_orders)
]
rider_locations = [
    (random.uniform(*latitude_range), random.uniform(*longitude_range)) for _ in range(n_riders)
]

# Create a folium map centered in Lucknow
lucknow_map = folium.Map(location=[26.85, 80.94], zoom_start=12)

# Add restaurant markers
for i, loc in enumerate(restaurant_locations):
    folium.Marker(location=loc, popup=f"Restaurant {i+1}", icon=folium.Icon(color='blue')).add_to(lucknow_map)

# Add delivery location markers
for i, loc in enumerate(delivery_locations):
    folium.Marker(location=loc, popup=f"Delivery {i+1}", icon=folium.Icon(color='green')).add_to(lucknow_map)

# Add rider markers
for i, loc in enumerate(rider_locations):
    folium.Marker(location=loc, popup=f"Rider {i+1}", icon=folium.Icon(color='orange')).add_to(lucknow_map)

# Simulate routes (connecting restaurants to delivery points)
for i in range(len(delivery_locations)):
    start = random.choice(restaurant_locations)
    end = delivery_locations[i]
    folium.PolyLine([start, end], color="red", weight=2.5, opacity=0.8).add_to(lucknow_map)

# Display the map
lucknow_map.save("delivery_map_with_riders.html")


#Code for OSRM Integration and Real-Time Updates

import folium
import requests
import random
from time import sleep

# Define Lucknow's geographical boundaries
latitude_range = (26.50, 27.17)
longitude_range = (80.50, 81.22)

# Generate random restaurant and delivery points
n_riders = 5  # Number of riders
n_orders = 10  # Number of orders
restaurant_locations = [
    (random.uniform(*latitude_range), random.uniform(*longitude_range)) for _ in range(3)
]
delivery_locations = [
    (random.uniform(*latitude_range), random.uniform(*longitude_range)) for _ in range(n_orders)
]
rider_locations = [
    (random.uniform(*latitude_range), random.uniform(*longitude_range)) for _ in range(n_riders)
]


# Function to get the route from OSRM

def decode_polyline(polyline):
    """Decode a Google Maps encoded polyline into a list of coordinates."""
    index, lat, lng, coordinates = 0, 0, 0, []

    while index < len(polyline):
        # Decode latitude
        shift, result = 0, 0
        while True:
            byte = ord(polyline[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lat = ~(result >> 1) if result & 1 else result >> 1
        lat += delta_lat

        # Decode longitude
        shift, result = 0, 0
        while True:
            byte = ord(polyline[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lng = ~(result >> 1) if result & 1 else result >> 1
        lng += delta_lng

        # Append decoded lat/lng to coordinates
        coordinates.append((lat / 1e5, lng / 1e5))

    return coordinates


def get_osrm_route(start, end):
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full"
    response = requests.get(osrm_url)
    if response.status_code == 200:
        data = response.json()
        if data['routes']:
            return data['routes'][0]['geometry']
    return None


# Create a folium map centered in Lucknow
lucknow_map = folium.Map(location=[26.85, 80.94], zoom_start=12)

# Add restaurant markers
for i, loc in enumerate(restaurant_locations):
    folium.Marker(location=loc, popup=f"Restaurant {i + 1}", icon=folium.Icon(color='blue')).add_to(lucknow_map)

# Add delivery location markers
for i, loc in enumerate(delivery_locations):
    folium.Marker(location=loc, popup=f"Delivery {i + 1}", icon=folium.Icon(color='green')).add_to(lucknow_map)

# Add rider markers and simulate routes
for i, rider_loc in enumerate(rider_locations):
    assigned_delivery = random.choice(delivery_locations)
    route_geometry = get_osrm_route(rider_loc, assigned_delivery)

    # Add rider marker
    folium.Marker(location=rider_loc, popup=f"Rider {i + 1}", icon=folium.Icon(color='orange')).add_to(lucknow_map)

    # Draw route if available
    if route_geometry:
        folium.PolyLine(
            locations=[tuple(reversed(coord)) for coord in decode_polyline(route_geometry)],  # Decode geometry
            color="red",
            weight=2.5,
            opacity=0.8
        ).add_to(lucknow_map)

# Save the map
lucknow_map.save("real_time_delivery_map.html")
print("Map saved as 'real_time_delivery_map.html'")

import folium
import random
import requests
from folium.plugins import MarkerCluster

# Sample data for locations (you should replace these with actual data)
restaurant_locations = [(26.85, 80.94), (26.87, 80.95), (26.89, 80.93)]  # Example restaurant locations
delivery_locations = [(26.86, 80.92), (26.88, 80.96), (26.84, 80.91)]  # Example delivery locations
rider_locations = [(26.83, 80.94), (26.85, 80.97), (26.87, 80.93)]  # Example rider locations


# Function to get the full route data from OSRM API (including distance)
def get_osrm_route(start, end):
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=polyline&steps=true"
    response = requests.get(osrm_url)
    if response.status_code == 200:
        data = response.json()
        if data['routes']:
            # Return the full route data, including polyline and distance
            return data['routes'][0]
    return None


# Function to decode polyline to coordinates
def decode_polyline(polyline):
    index, lat, lng, coordinates = 0, 0, 0, []
    while index < len(polyline):
        shift, result = 0, 0
        while True:
            byte = ord(polyline[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lat = ~(result >> 1) if result & 1 else result >> 1
        lat += delta_lat

        shift, result = 0, 0
        while True:
            byte = ord(polyline[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lng = ~(result >> 1) if result & 1 else result >> 1
        lng += delta_lng

        coordinates.append((lat / 1e5, lng / 1e5))
    return coordinates


# Create a folium map centered in Lucknow
lucknow_map = folium.Map(location=[26.85, 80.94], zoom_start=12)

# Add a marker cluster for restaurants and deliveries
marker_cluster = MarkerCluster().add_to(lucknow_map)

# Add restaurant markers with enhanced popups
for i, loc in enumerate(restaurant_locations):
    folium.Marker(
        location=loc,
        popup=f"<b>Restaurant {i + 1}</b><br>Location: {loc[0]}, {loc[1]}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Add delivery location markers with enhanced popups
for i, loc in enumerate(delivery_locations):
    folium.Marker(
        location=loc,
        popup=f"<b>Delivery {i + 1}</b><br>Location: {loc[0]}, {loc[1]}",
        icon=folium.Icon(color='green', icon='gift')
    ).add_to(marker_cluster)

# Add rider markers with enhanced popups and simulate routes
for i, rider_loc in enumerate(rider_locations):
    assigned_delivery = random.choice(delivery_locations)
    route_data = get_osrm_route(rider_loc, assigned_delivery)

    # Add rider marker with a popup
    folium.Marker(
        location=rider_loc,
        popup=f"<b>Rider {i + 1}</b><br>Location: {rider_loc[0]}, {rider_loc[1]}",
        icon=folium.Icon(color='orange', icon='bicycle')
    ).add_to(marker_cluster)

    # Draw route if available
    if route_data:
        route_geometry = route_data['geometry']
        # Decode polyline to get coordinates
        route_coordinates = decode_polyline(route_geometry)
        # Add route polyline to map
        folium.PolyLine(
            locations=[tuple(reversed(coord)) for coord in route_coordinates],  # Decode geometry
            color="red",
            weight=2.5,
            opacity=0.8
        ).add_to(lucknow_map)

        # Get route distance (in meters) and convert to kilometers
        route_distance = route_data['legs'][0]['distance'] / 1000  # Distance in km
        folium.Marker(
            location=rider_loc,
            popup=f"Distance to Delivery {i + 1}: {route_distance:.2f} km",
            icon=folium.Icon(color='purple', icon='info-sign')
        ).add_to(lucknow_map)

# Save the map to an HTML file
lucknow_map.save("enhanced_real_time_delivery_map.html")
print("Map saved as 'enhanced_real_time_delivery_map.html'")

import folium
import random
import requests
from folium.plugins import MarkerCluster

# Sample data for locations (you should replace these with actual data)
restaurant_locations = [(26.85, 80.94), (26.87, 80.95), (26.89, 80.93)]  # Example restaurant locations
delivery_locations = [(26.86, 80.92), (26.88, 80.96), (26.84, 80.91)]  # Example delivery locations
rider_locations = [(26.83, 80.94), (26.85, 80.97), (26.87, 80.93)]  # Example rider locations


# Function to get the full route data from OSRM API (including distance)
def get_osrm_route(start, end):
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=polyline&steps=true"
    response = requests.get(osrm_url)
    if response.status_code == 200:
        data = response.json()
        if data['routes']:
            # Return the full route data, including polyline and distance
            return data['routes'][0]
    return None


# Function to decode polyline to coordinates
def decode_polyline(polyline):
    index, lat, lng, coordinates = 0, 0, 0, []
    while index < len(polyline):
        shift, result = 0, 0
        while True:
            byte = ord(polyline[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lat = ~(result >> 1) if result & 1 else result >> 1
        lat += delta_lat

        shift, result = 0, 0
        while True:
            byte = ord(polyline[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lng = ~(result >> 1) if result & 1 else result >> 1
        lng += delta_lng

        coordinates.append((lat / 1e5, lng / 1e5))
    return coordinates


# Create a folium map centered in Lucknow
lucknow_map = folium.Map(location=[26.85, 80.94], zoom_start=12)

# Add a marker cluster for restaurants and deliveries
marker_cluster = MarkerCluster().add_to(lucknow_map)

# Add restaurant markers with enhanced popups
for i, loc in enumerate(restaurant_locations):
    folium.Marker(
        location=loc,
        popup=f"<b>Restaurant {i + 1}</b><br>Location: {loc[0]}, {loc[1]}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Add delivery location markers with enhanced popups
for i, loc in enumerate(delivery_locations):
    folium.Marker(
        location=loc,
        popup=f"<b>Delivery {i + 1}</b><br>Location: {loc[0]}, {loc[1]}",
        icon=folium.Icon(color='green', icon='gift')
    ).add_to(marker_cluster)

# Add rider markers with enhanced popups and simulate routes
for i, rider_loc in enumerate(rider_locations):
    assigned_restaurant = random.choice(restaurant_locations)
    assigned_delivery = random.choice(delivery_locations)

    # Step 1: Route from Rider to Restaurant
    route_to_restaurant = get_osrm_route(rider_loc, assigned_restaurant)

    # Step 2: Route from Restaurant to Delivery
    route_to_delivery = get_osrm_route(assigned_restaurant, assigned_delivery)

    # Add rider marker with a popup
    folium.Marker(
        location=rider_loc,
        popup=f"<b>Rider {i + 1}</b><br>Location: {rider_loc[0]}, {rider_loc[1]}",
        icon=folium.Icon(color='orange', icon='bicycle')
    ).add_to(marker_cluster)

    # Draw Route from Rider to Restaurant
    if route_to_restaurant:
        route_geometry = route_to_restaurant['geometry']
        route_coordinates = decode_polyline(route_geometry)
        folium.PolyLine(
            locations=[tuple(reversed(coord)) for coord in route_coordinates],  # Decode geometry
            color="blue",
            weight=2.5,
            opacity=0.8,
            popup=f"Rider {i + 1} to Restaurant"
        ).add_to(lucknow_map)

    # Draw Route from Restaurant to Delivery
    if route_to_delivery:
        route_geometry = route_to_delivery['geometry']
        route_coordinates = decode_polyline(route_geometry)
        folium.PolyLine(
            locations=[tuple(reversed(coord)) for coord in route_coordinates],  # Decode geometry
            color="red",
            weight=2.5,
            opacity=0.8,
            popup=f"Restaurant to Delivery for Rider {i + 1}"
        ).add_to(lucknow_map)

    # Add the final delivery marker
    folium.Marker(
        location=assigned_delivery,
        popup=f"<b>Delivery {i + 1}</b><br>Location: {assigned_delivery[0]}, {assigned_delivery[1]}",
        icon=folium.Icon(color='green', icon='gift')
    ).add_to(marker_cluster)

# Save the map to an HTML file
lucknow_map.save("rider_delivery_routes_map.html")
print("Map saved as 'rider_delivery_routes_map.html'")

import folium
import random
import requests
from folium.plugins import MarkerCluster

# Sample data for locations (replace these with actual data)
restaurant_locations = [(26.85, 80.94), (26.87, 80.95), (26.89, 80.93)]  # Example restaurant locations
delivery_locations = [(26.86, 80.92), (26.88, 80.96), (26.84, 80.91)]  # Example delivery locations
rider_locations = [(26.83, 80.94), (26.85, 80.97), (26.87, 80.93)]  # Example rider locations


# Function to get the full route data from OSRM API (including distance)
def get_osrm_route(start, end):
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=polyline&steps=true"
    response = requests.get(osrm_url)
    if response.status_code == 200:
        data = response.json()
        if data['routes']:
            return data['routes'][0]
    return None


# Function to decode polyline to coordinates
def decode_polyline(polyline):
    index, lat, lng, coordinates = 0, 0, 0, []
    while index < len(polyline):
        shift, result = 0, 0
        while True:
            byte = ord(polyline[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lat = ~(result >> 1) if result & 1 else result >> 1
        lat += delta_lat

        shift, result = 0, 0
        while True:
            byte = ord(polyline[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lng = ~(result >> 1) if result & 1 else result >> 1
        lng += delta_lng

        coordinates.append((lat / 1e5, lng / 1e5))
    return coordinates


# Create a folium map centered in Lucknow
lucknow_map = folium.Map(location=[26.85, 80.94], zoom_start=12)

# Add a marker cluster for restaurants and deliveries
marker_cluster = MarkerCluster().add_to(lucknow_map)

# Add restaurant markers with enhanced popups
for i, loc in enumerate(restaurant_locations):
    folium.Marker(
        location=loc,
        popup=f"<b>Restaurant {i + 1}</b><br>Location: {loc[0]}, {loc[1]}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Add delivery location markers with enhanced popups
for i, loc in enumerate(delivery_locations):
    folium.Marker(
        location=loc,
        popup=f"<b>Delivery {i + 1}</b><br>Location: {loc[0]}, {loc[1]}",
        icon=folium.Icon(color='green', icon='gift')
    ).add_to(marker_cluster)

# Add rider markers with enhanced popups and simulate routes
for i, rider_loc in enumerate(rider_locations):
    assigned_restaurant = random.choice(restaurant_locations)
    assigned_delivery = random.choice(delivery_locations)

    # Step 1: Route from Rider to Restaurant
    route_to_restaurant = get_osrm_route(rider_loc, assigned_restaurant)

    # Step 2: Route from Restaurant to Delivery
    route_to_delivery = get_osrm_route(assigned_restaurant, assigned_delivery)

    # Add rider marker with a popup
    folium.Marker(
        location=rider_loc,
        popup=f"<b>Rider {i + 1}</b><br>Location: {rider_loc[0]}, {rider_loc[1]}",
        icon=folium.Icon(color='orange', icon='bicycle')
    ).add_to(marker_cluster)

    # Simulating reinforcement learning routes
    # RL-based optimal routes (simulated as green)
    if route_to_restaurant:
        route_geometry = route_to_restaurant['geometry']
        route_coordinates = decode_polyline(route_geometry)
        folium.PolyLine(
            locations=[tuple(reversed(coord)) for coord in route_coordinates],  # Decode geometry
            color="green",  # Simulating optimal path (RL)
            weight=4,
            opacity=0.8,
            popup=f"Rider {i + 1} to Restaurant (RL Optimal)",
            tooltip="RL Optimal Route to Restaurant"
        ).add_to(lucknow_map)

    if route_to_delivery:
        route_geometry = route_to_delivery['geometry']
        route_coordinates = decode_polyline(route_geometry)
        folium.PolyLine(
            locations=[tuple(reversed(coord)) for coord in route_coordinates],  # Decode geometry
            color="green",  # Simulating optimal path (RL)
            weight=4,
            opacity=0.8,
            popup=f"Restaurant to Delivery for Rider {i + 1} (RL Optimal)",
            tooltip="RL Optimal Route to Delivery"
        ).add_to(lucknow_map)

    # Add the final delivery marker
    folium.Marker(
        location=assigned_delivery,
        popup=f"<b>Delivery {i + 1}</b><br>Location: {assigned_delivery[0]}, {assigned_delivery[1]}",
        icon=folium.Icon(color='green', icon='gift')
    ).add_to(marker_cluster)

# Add a legend to explain the colors
legend_html = """
<div style="position: fixed; bottom: 20px; left: 20px; width: 150px; height: 120px; background-color: white; opacity: 0.7; padding: 10px;">
    <b>Legend:</b><br>
    <b>Green:</b> Optimal RL Path<br>
    <b>Orange:</b> Rider's Location<br>
    <b>Blue:</b> Restaurant<br>
    <b>Green:</b> Delivery Location<br>
</div>
"""
lucknow_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
lucknow_map.save("rider_delivery_routes_with_rl_map.html")
print("Map saved as 'rider_delivery_routes_with_rl_map.html'")

import folium
import random
import openrouteservice

# Sample data for locations (replace these with actual data)
restaurant_locations = [(26.84813819912435, 80.94072639861156)]  # Example restaurant locations
delivery_locations = [(26.84837231425606, 80.93380151349686)]  # Example delivery locations
rider_locations = [(26.847751023826472, 80.94066141815843)]  # Example rider locations

# OpenRouteService API key (replace with your key)
ORS_API_KEY = '5b3ce3597851110001cf6248b185d93ead8346d1adbd5587068dab38'

# Initialize OpenRouteService client
client = openrouteservice.Client(key=ORS_API_KEY)


# Function to get route from ORS API
def get_ors_route(start, end):
    # Make request to ORS API for the route
    route = client.directions(
        coordinates=[start, end],  # Start and end coordinates
        profile='driving-car',  # You can change this for different profiles like walking, cycling, etc.
        format='geojson'
    )
    return route


# Function to decode coordinates from geojson
def decode_geojson(route_geojson):
    return [(coord[1], coord[0]) for coord in route_geojson['features'][0]['geometry']['coordinates']]


# Create a folium map centered in Lucknow
lucknow_map = folium.Map(location=[26.85, 80.94], zoom_start=12)

# Add restaurant markers with popups
for i, loc in enumerate(restaurant_locations):
    folium.Marker(
        location=loc,
        popup=f"<b>Restaurant {i + 1}</b><br>Location: {loc[0]}, {loc[1]}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(lucknow_map)

# Add delivery location markers with popups
for i, loc in enumerate(delivery_locations):
    folium.Marker(
        location=loc,
        popup=f"<b>Delivery {i + 1}</b><br>Location: {loc[0]}, {loc[1]}",
        icon=folium.Icon(color='green', icon='gift')
    ).add_to(lucknow_map)

# Add rider markers with popups and simulate routes
for i, rider_loc in enumerate(rider_locations):
    assigned_restaurant = random.choice(restaurant_locations)
    assigned_delivery = random.choice(delivery_locations)

    # Step 1: Route from Rider to Restaurant (ORS API call)
    route_to_restaurant = get_ors_route(rider_loc, assigned_restaurant)

    # Step 2: Route from Restaurant to Delivery (ORS API call)
    route_to_delivery = get_ors_route(assigned_restaurant, assigned_delivery)

    # Add rider marker with a popup
    folium.Marker(
        location=rider_loc,
        popup=f"<b>Rider {i + 1}</b><br>Location: {rider_loc[0]}, {rider_loc[1]}",
        icon=folium.Icon(color='orange', icon='bicycle')
    ).add_to(lucknow_map)

    # Draw the route from Rider to Restaurant (Blue Line)
    if route_to_restaurant:
        route_coordinates = decode_geojson(route_to_restaurant)
        folium.PolyLine(
            locations=route_coordinates,  # Route from rider to restaurant
            color="blue",  # Showing route to restaurant
            weight=4,
            opacity=0.8,
            popup=f"Rider {i + 1} to Restaurant",
            tooltip="Route from Rider to Restaurant"
        ).add_to(lucknow_map)

    # Draw the route from Restaurant to Delivery (Red Line)
    if route_to_delivery:
        route_coordinates = decode_geojson(route_to_delivery)
        folium.PolyLine(
            locations=route_coordinates,  # Route from restaurant to delivery
            color="red",  # Showing route to delivery
            weight=4,
            opacity=0.8,
            popup=f"Restaurant to Delivery for Rider {i + 1}",
            tooltip="Route from Restaurant to Delivery"
        ).add_to(lucknow_map)

    # Add the final delivery marker
    folium.Marker(
        location=assigned_delivery,
        popup=f"<b>Delivery {i + 1}</b><br>Location: {assigned_delivery[0]}, {assigned_delivery[1]}",
        icon=folium.Icon(color='green', icon='gift')
    ).add_to(lucknow_map)

# Save the map to an HTML file
lucknow_map.save("rider_delivery_routes_map_with_ors.html")
print("Map saved as 'rider_delivery_routes_map_with_ors.html'")
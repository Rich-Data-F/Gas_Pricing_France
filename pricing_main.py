import xml.etree.ElementTree as ET
import pandas as pd
import duckdb
from geopy.geocoders import Nominatim
import folium
import streamlit as st
from streamlit_folium import folium_static
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
import numpy as np
import os
from datetime import datetime
import locale
import zipfile
import io
import requests

from pricing_definitions import *

folder_path='./Data/'
file_name='PrixCarburants_instantane.xml'
get_file_info(folder_path, file_name)
import locale
# Set locale to French
try:
    locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_TIME, 'C')  # or 'en_US.UTF-8' if available
#Combine the date and time strings

file_info = get_file_info(folder_path, file_name)
# Parse the date and time strings back into a datetime object
update_datetime = datetime.strptime(f"{file_info['creation_date']} {file_info['creation_time']}", "%Y-%m-%d %H:%M:%S")
# Format it as desired
formatted_datetime = update_datetime.strftime("%d %B %Y à %Hh%M")
print(f"Les données de prix de carburant ont été mises à jour le {formatted_datetime}. Pressez le bouton ci-dessous pour une mise à jour")

df=pd.read_xml('Data/PrixCarburants_instantane.xml', 
                 xpath='.//pdv',
                 encoding='ISO-8859-1',
                 dtype=str)
df.head()

# Use the function to read your XML file
df = parse_xml('Data/PrixCarburants_instantane.xml')
df.head()

df['toilettes_presentes'] = df['services'].apply(lambda services: 'Toilettes publiques' in services if isinstance(services, list) else False)
df.head(1)
df.size
sample_df = df.sample(n=5000, random_state=42)
print(sample_df.describe())
sample_df.dtypes
df.columns
df.info()

gas_price_columns = [col for col in df.columns if col.startswith('prix_')]
print("Gas price columns:", gas_price_columns)
#list comprehension to remove the text before the _
gas_types = [item.split("_", 1)[-1] for item in gas_price_columns]
gas_types

# Usage
df = process_gas_prices(df)

# Display the first few rows to verify
print(df[['id'] + [col for col in df.columns if col.endswith('_price')] + ['last_updated']].head())
df['department']=df['cp'].str[:2]
df.head()
df.info()
df.value_counts(subset=['last_updated'])
len(df['last_updated'].value_counts())
df.head()

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
# Now perform the division
df['latitude'] = df['latitude'] / 100000
df['longitude'] = df['longitude'] / 100000
# Optional: Remove any rows where latitude or longitude is NaN
df = df.dropna(subset=['latitude', 'longitude'])
# Print the first few rows to verify the conversion
print(df[['latitude', 'longitude']].head())
# Print the data types of these columns
print(df[['latitude', 'longitude']].dtypes)

# Main program
print("Please enter the address details:")
try:
    user_address, user_postal_code = prompt_for_address()
    print(f"\nSearching for coordinates of: {user_address}")
    result = get_lat_long(user_address)
    if result:
        user_latitude, user_longitude = result
        print(f"Latitude: {user_latitude}, Longitude: {user_longitude}")
    else:
        print("Could not find coordinates for the given address")
except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

selected_gas_type = select_gas_type(gas_types)
print(f"\nYou selected: {selected_gas_type}")

file_path='Data/French_adjacent_departments.txt'
adjacent_deps=read_adjacent_departments(file_path)

relevant_depts=get_relevant_departments(user_postal_code, adjacent_deps)
filtered_df=filter_gas_stations_df(df, relevant_depts)
filtered_df['latitude'] = pd.to_numeric(filtered_df['latitude'], errors='coerce')
filtered_df['longitude'] = pd.to_numeric(filtered_df['longitude'], errors='coerce')

m = create_map(user_address, user_latitude, user_longitude, filtered_df, selected_gas_type)
folium_static(m)
## filtered_df unknown to be replaced

print(filtered_df.shape)
print(filtered_df.columns)
print(filtered_df.head())
print(filtered_df.columns.unique())

m = create_map(user_address, user_latitude, user_longitude, filtered_df, selected_gas_type)
print(type(m))
folium_static(m)
output_dir = 'Data/user_specific'
os.makedirs(output_dir, exist_ok=True)
# Generate a timestamp for a unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Save the map as an HTML file
map_filename = f'map_{timestamp}.html'
map_path = os.path.join(output_dir, map_filename)
m.save(map_path)
print(f"Map saved to: {map_path}")

# oversimplistic calculation 
'''filtered_df['distance'] = ((filtered_df['latitude'] - user_latitude)**2 + 
                                        (filtered_df['longitude'] - user_longitude)**2)**0.5
closest_stations = filtered_df.nsmallest(10, 'distance')
st.table(closest_stations[['name', 'distance', f'price_{selected_gas_type}', 'last_update']])'''

# Calculate distances
filtered_df.loc[:,'distance'] = filtered_df.apply(lambda row: haversine_distance(user_latitude, user_longitude, 
                                                                           row['latitude'], row['longitude']), axis=1)

# Get the 10 closest stations
closest_stations = filtered_df.nsmallest(10, 'distance')
print(closest_stations.shape)
print(closest_stations.columns)

# List of columns you want to display
columns_to_display = ['id', 'distance in km', f'{selected_gas_type}_price', 'last_updated']

# Filter to only include columns that actually exist in the DataFrame
existing_columns = [col for col in columns_to_display if col in closest_stations.columns]

# Display the table with only the existing columns
st.table(closest_stations[existing_columns])

if not existing_columns:
    st.error("No matching columns found in the data.")
elif len(existing_columns) < len(columns_to_display):
    st.warning(f"Some columns were not found: {set(columns_to_display) - set(existing_columns)}")

# Create a directory to store the files
output_dir = 'Data/user_specific'
os.makedirs(output_dir, exist_ok=True)

# Generate a timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save filtered_df to CSV
filtered_df_filename = f'filtered_df_{timestamp}.csv'
filtered_df_path = os.path.join(output_dir, filtered_df_filename)
filtered_df.to_csv(filtered_df_path, index=False)
print(f"Filtered DataFrame saved to: {filtered_df_path}")

# Save closest_stations to CSV
closest_stations_filename = f'closest_stations_{timestamp}.csv'
closest_stations_path = os.path.join(output_dir, closest_stations_filename)
closest_stations.to_csv(closest_stations_path, index=False)
print(f"Closest stations DataFrame saved to: {closest_stations_path}")

# Display the results (assuming you're using Streamlit)
#st.table(closest_stations[['name', 'distance', f'price_{selected_gas_type}', 'last_update']])


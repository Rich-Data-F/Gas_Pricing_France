import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
import os
from datetime import datetime

from pricing_definitions import *

st.set_page_config(page_title="Gas Price Finder", page_icon="⛽")

st.title("Gas Price Finder in France")

# Input fields for address
street = st.text_input("Street Address", "15, rue de Vaugirard")
city = st.text_input("City", "Paris")
zipcode = st.text_input("Zipcode", "75000")
country = st.text_input("Country", "France", disabled=True)

# Combine address components
user_address = f"{street}, {zipcode}, {city}, {country}"

# Function to get coordinates
def get_coordinates(address):
    geolocator = Nominatim(user_agent="gas_price_finder")
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            st.error(f"Address not found: {address}")
            return None
    except (GeocoderServiceError, GeocoderTimedOut) as e:
        st.error(f"Error occurred: {e}")
        return None

# Get user coordinates
if st.button("Find Gas Stations"):
    result = get_coordinates(user_address)
    if result:
        user_latitude, user_longitude = result
        st.success(f"Coordinates found: Latitude {user_latitude}, Longitude {user_longitude}")

        # Load and process data
        df = parse_xml('Data/PrixCarburants_instantane.xml')
        df = process_gas_prices(df)

        # Process departments
        df['department'] = df['cp'].str[:2]
        df['toilettes_presentes'] = df['services'].apply(lambda services: 'Toilettes publiques' in services if isinstance(services, list) else False)
        # conversion of coordinates in proper format 
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        # Division to include unit expected by geopy and enabling to properly calculate distance.
        df['latitude'] = df['latitude'] / 100000
        df['longitude'] = df['longitude'] / 100000
        # Optional: Remove any rows where latitude or longitude is NaN
        df = df.dropna(subset=['latitude', 'longitude'])
        # Print the first few rows to verify the conversion
        print(df[['latitude', 'longitude']].head())
        # Print the data types of these columns
        print(df[['latitude', 'longitude']].dtypes)
        df['autoroute']=df['pop'].apply(lambda pop: pop == 'A' if isinstance(pop, str) else False)
        file_path = 'Data/French_adjacent_departments.txt'
        adjacent_deps = read_adjacent_departments(file_path)
        relevant_depts = get_relevant_departments(zipcode, adjacent_deps)
        filtered_df = filter_gas_stations_df(df, relevant_depts)
        st.dataframe(filtered_df)
        # Get gas types
        gas_price_columns = [col for col in df.columns if col.startswith('prix_')]
        print("Gas price columns:", gas_price_columns)
        #list comprehension to remove the text before the _
        gas_types = [item.split("_", 1)[-1] for item in gas_price_columns]
        print(gas_types)
        #default_index = gas_types.index('Gazole') if 'Gazole' in gas_types else 0
        # Select gas type
        #selected_gas_type = st.selectbox("Select Gas Type", gas_types,index=default_index)
        #selected_gas_type = st.radio("Select Gas Type", gas_types, index=gas_types.index('Gazole') if 'Gazole' in gas_types else 0)
        selected_gas_type = 'Gazole'
        if selected_gas_type:
            st.success(f"Vous avez choisi le carburant {selected_gas_type}")
        # Calculate distances and get closest stations
            filtered_df['distance'] = filtered_df.apply(lambda row: haversine_distance(user_latitude, user_longitude, 
                                                                                   row['latitude'], row['longitude']), axis=1)
            closest_stations = filtered_df.nsmallest(100, 'distance')
            # Create map
            m = create_map(user_address, user_latitude, user_longitude, filtered_df, selected_gas_type)
            folium_static(m)
            # Display closest stations
            # Check if 'distance' column exists and is not empty
            if 'distance' in closest_stations.columns and not closest_stations['distance'].empty:
                # Get min and max values for the distance column
                min_distance = float(closest_stations['distance'].min())
                max_distance = float(closest_stations['distance'].max())
                # Create a range slider for distance
                distance_range = st.slider(
                    "Select Distance Range (km)",
                    min_value=min_distance,
                    max_value=max_distance,
                    value=(min_distance, max_distance)
                )
                # Filter the DataFrame based on the selected distance range
                filtered_stations = closest_stations[
                    (closest_stations['distance'] >= distance_range[0]) &
                    (closest_stations['distance'] <= distance_range[1])
                ]
                # Rename columns for display clarity
                filtered_stations = filtered_stations.rename(columns={
                    'distance': 'distance en km',
                    'autoroute': "sur l'autoroute"
                })
                # Remove rows with NaN values for the selected gas type price
                filtered_stations = filtered_stations[filtered_stations[f'{selected_gas_type}_price'].notna()]
                # Display the filtered DataFrame
                if not filtered_stations.empty:
                    st.dataframe(filtered_stations[['id', 'adresse', 'ville', 'distance en km', f'{selected_gas_type}_price', 'last_updated', "sur l'autoroute"]])
                else:
                    st.warning("No stations found within the selected distance range.")
                # Display the number of stations shown
                st.info(f"Showing {len(filtered_stations)} stations")
        else:
            st.error("Please select a gas type to continue")
    else:
        st.error("Could not find coordinates for the given address")

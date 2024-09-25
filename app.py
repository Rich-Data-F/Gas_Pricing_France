import streamlit as st
import sys
import pandas as pd
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
import os
from datetime import datetime, timedelta
import requests
import zipfile
import io
import logging
import openrouteservice as ors
from dotenv import load_dotenv
from cryptography.fernet import Fernet

from pricing_definitions import *

# initialisation of session state entities
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = None
if 'user_latitude' not in st.session_state:
    st.session_state['user_latitude'] = None
if 'user_longitude' not in st.session_state:
    st.session_state['user_longitude'] = None
map_path=''
if map_path not in st.session_state:
    st.session_state.map_path=''

st.set_page_config(page_title="Gas Station and Best Price locator", page_icon="⛽", layout="wide")

st.markdown("""
    <style>
    .stbutton > button {
        background-color:  #FFD580;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Get the API key
#api_key = get_api_key()
api_key=st.secrets["OPENROUTE_API_KEY"]

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sidebar for user inputs
st.sidebar.title("Gas Station Finder Options")

# 1. Gas type selection (mandatory)
gas_types = ['Gazole', 'SP95', 'SP98', 'GPLc', 'E10', 'E85']  # Add all available gas types
selected_gas_type = st.sidebar.selectbox("Select Gas Type", gas_types, index=gas_types.index('Gazole'))

# Optional inputs
st.sidebar.subheader("Optional Filters")

# a. Outstanding autonomy
autonomy = st.sidebar.slider("Remaining Autonomy (km)", min_value=5, max_value=125, value=45, step=1)

# b. Toilets required
#toilets_required = st.sidebar.radio("Toilets Required", [True, False], index=0)

# c. Spare time to save money
#spare_time = st.sidebar.radio("Willing to spend time to save money?", ['Yes', 'No'], index=0)

# d. Autoroutes allowed
#autoroutes_allowed = st.sidebar.radio("Autoroutes Allowed", ["Yes", "No"], index=0)

# e. Vehicle consumption
consumption = st.sidebar.number_input("Vehicle Consumption (L/100km)", value=6.0, min_value=0.5, step=0.5)

# f. Gas tank total volume
tank_volume = st.sidebar.slider("Gas Tank Total Volume (L)", min_value=5, max_value=80, value=45, step=1)

# g. Current gas tank volume left (fraction)
tank_left = st.sidebar.slider("Current Gas Tank Volume Left (fraction)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

# h. radius search size in kms
radius_search = st.sidebar.slider("Size of radius search (km)", min_value=1, max_value=(int(autonomy)-5), value=50, step=2)

st.sidebar.write("The recommendation is limited to the 40 closest stations to the selected address, which may prevail on the above-defined distance and criteria")

# Main content
st.title("Gas Station and Best Price locator")

folder_path='./Data/'
file_name='PrixCarburants_instantane.xml'

file_info = get_file_info(folder_path, file_name)
# Parse the date and time strings back into a datetime object
update_datetime = datetime.strptime(f"{file_info['creation_date']} {file_info['creation_time']}", "%Y-%m-%d %H:%M:%S")
# Format it as desired
formatted_datetime = update_datetime.strftime("%d %B %Y à %Hh%M")
st.write(f"Price information have been last updated on {formatted_datetime}. Please press the 'refresh' button below for a newer update")
st.write(f"source: https://www.prix-carburants.gouv.fr/rubrique/opendata/, limited to metropolitan France")

refresh=st.button(help='A few seconds will be required to get refreshed data', label="Get refreshed data")

if refresh:
    # Step 1: Download the ZIP file from the HTML address
    url = "https://donnees.roulez-eco.fr/opendata/instantane"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    # Step 2: Extract the XML file from the ZIP
    xml_content = zip_file.read('PrixCarburants_instantane.xml')
    # Step 3: Save the XML content to the Data folder
    data_folder = "Data"  # Adjust this path as needed
    file_path = os.path.join(data_folder, "PrixCarburants_instantane.xml")
    with open(file_path, "wb") as f:
        f.write(xml_content)
    # Step 4: Read the XML content using pd.read_xml()
    df = pd.read_xml(file_path, xpath='.//pdv',
                    encoding='ISO-8859-1',
                    dtype=str)
    # Step 5: display the refreshed date
    file_info = get_file_info(folder_path, file_name)
    # Parse the date and time strings back into a datetime object
    update_datetime = datetime.strptime(f"{file_info['creation_date']} {file_info['creation_time']}", "%Y-%m-%d %H:%M:%S")
    # Format it as desired
    formatted_datetime = update_datetime.strftime("%d %B %Y à %Hh%M")
    st.write(f"Les données de prix de carburant ont été mises à jour le {formatted_datetime}. Pressez le bouton ci-dessous pour une mise à jour")

# Address inputs
col1, col2, col3 = st.columns(3)
with col1:
    street = st.text_input("Street Address", "rue des clés")
with col2:
    city = st.text_input("City", "Thônes")
with col3:
    zipcode = st.text_input("Zipcode or at least department number", "74")

#country = st.text_input("Country", "France", disabled=True)
country = "France"
# Combine address components
user_address = f"{street}, {zipcode}, {city}, {country}"

# Get user coordinates
st.write("A more precise and tailored recommendation can be made using extra information provided in left column. The 'super recommendation' button can be actioned once the first 'Find Gas Stations' step is actioned")
if st.button("Find Gas Stations"):
    result = get_coordinates(user_address)
    if result:
        user_latitude, user_longitude = result
        st.session_state['user_latitude'] = user_latitude
        st.session_state['user_longitude'] = user_longitude
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
        filtered_df = filter_gas_stations_df(df, relevant_depts).copy()
        st.session_state['filtered_df']=filtered_df
        st.dataframe(filtered_df)
        # Get gas types
        gas_price_columns = [col for col in df.columns if col.startswith('prix_')]
        print("Gas price columns:", gas_price_columns)
        #list comprehension to remove the text before the _
        gas_types = [item.split("_", 1)[-1] for item in gas_price_columns]
        print(gas_types)
        #default_index = gas_types.index('Gazole') if 'Gazole' in gas_types else 0
        if selected_gas_type:
            st.success(f"Vous avez choisi le carburant {selected_gas_type}")
        # Calculate distances and get closest stations
            filtered_df['distance'] = filtered_df.apply(lambda row: haversine_distance(user_latitude, user_longitude, 
                                                                                   row['latitude'], row['longitude']), axis=1)
            closest_stations = filtered_df.nsmallest(100, 'distance')
            # Create map
            m = create_map(user_address, user_latitude, user_longitude, filtered_df, selected_gas_type)
            folium_static(m, width=1200, height=800)
            output_dir = 'Data/user_specific'
            os.makedirs(output_dir, exist_ok=True)
            # Generate a timestamp for a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Save the map as an HTML file
            map_filename = f'map_{timestamp}.html'
            map_path = os.path.join(output_dir, map_filename)
            st.session_state.map_path=map_path
            m.save(st.session_state.map_path)
            print(f"Map saved to: {st.session_state.map_path}")
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

if st.button("Super Recommendation"):
    st.toast("Processing calculations... Should take less than a minute ...")
    if st.session_state['filtered_df'] is not None and st.session_state['user_latitude'] is not None and st.session_state['user_longitude'] is not None:
        # Perform the advanced filtering and calculations# Perform the advanced filtering and calculations
        super_filters = st.session_state['filtered_df'][
            (st.session_state['filtered_df']['distance']<=radius_search) &
            (st.session_state['filtered_df'][f'{selected_gas_type}_price'].notna())
            ].copy()
        user_latitude = st.session_state['user_latitude']
        user_longitude = st.session_state['user_longitude']
        if super_filters.empty:
            st.warning(f"No stations found within {radius_search} kms. Please try increasing the search radius.")
        else:
            #setup logging
            logging.basicConfig(level=logging.INFO)
            # ii. Calculate actual distance to station (adts) using a routing library
            def calculate_route_distance(start_coords, end_coords):
                try:
                    # Use the API key in your Streamlit app
                    client = ors.Client(key=api_key)  # Free API key
                    start_lon, start_lat = float(start_coords[0]), float(start_coords[1])
                    end_lon, end_lat = float(end_coords[0]), float(end_coords[1])
                    logging.info(f"Start coordinates: {start_coords}")
                    logging.info(f"End coordinates: {end_coords}")
                    coords = [[start_lon, start_lat], [end_lon, end_lat]]
                    print(coords)
                    route = client.directions(
                        coordinates=coords,
                        profile='driving-car'
                        )
                    raw_distance = route['routes'][0]['summary']['distance']
                    distance = round(raw_distance / 1000, 3)
                    print(f"Distance: {distance} km")
                    logging.info(f"ors_distance:{distance}")
                    return distance
                except Exception as e:
                    logging.error(f"Error calculating route: {str(e)}", exc_info=True)
                    return None
            # Sort the DataFrame by the 'distance' column
            super_filters = super_filters.sort_values('distance')
            # Create a temporary Series to hold the results
            temp_adts = pd.Series(index=super_filters.index)
            # Apply the function to the first 40 rows
            temp_adts.iloc[:40] = super_filters.iloc[:40].apply(
                lambda row: calculate_route_distance(
                    (user_longitude, user_latitude), 
                    (row['longitude'], row['latitude'])
                ), 
                axis=1
            )
            # Assign the results back to the 'adts' column
            super_filters['adts'] = temp_adts
            # If you want to fill the remaining rows with a placeholder value (e.g., NaN or -1)
            super_filters['adts'] = super_filters['adts'].fillna(np.nan)  # or use .fillna(np.nan) for NaN
            # super_filters['adts'] = super_filters.apply(lambda row: calculate_route_distance((user_longitude, user_latitude), (row['longitude'], row['latitude'])), axis=1)                        
            super_filters = super_filters.dropna(subset=[f'{selected_gas_type}_price']) # Remove rows where selected gas_type is not available
            super_filters = super_filters.dropna(subset=['adts'])  # Remove rows where route calculation failed
            # iii. Calculate costs and time to reach station
            # average price of the gas type in the 50 closest stations
            apgt50 = st.session_state['filtered_df'].nsmallest(50, 'distance')[f'{selected_gas_type}_price'].mean()  
            # cost to get to the station
            super_filters['cost_to_reach'] = super_filters['adts'] * consumption / 100 * apgt50
            super_filters['time_to_reach'] = super_filters['adts'].apply(lambda x: format_timedelta(timedelta(hours=x/50)))
            # iv. Calculate summary
            super_filters['total_cost'] = (tank_volume * (1 - tank_left)) * super_filters[f'{selected_gas_type}_price'] + \
                                        2*super_filters['adts'] * consumption / 100 * apgt50
            # Print some diagnostics
            print(f"Number of failed calculations: {(super_filters['adts'] == -1).sum()}")
            print(super_filters['adts'].describe())
#            super_filters['adts'] = super_filters.apply(lambda row: calculate_route_distance((user_latitude, user_longitude), (row['latitude'], row['longitude'])), axis=1)
            # Display the average price of gas type in 50 closest stations
            st.write(f"Average price of {selected_gas_type} in 50 closest stations: {apgt50:.2f} €/L")
            # Display the super_filters dataframe
            additional_columns_to_drop=['horaires','pop']
            gas_columns_to_drop = [f"{gas}_price" for gas in gas_types if gas != selected_gas_type]
            print(gas_columns_to_drop)
            columns_to_drop = gas_columns_to_drop + additional_columns_to_drop
            super_filters = super_filters.drop(columns=columns_to_drop)
            super_filters_sorted = super_filters.sort_values(by=['total_cost', 'time_to_reach'], ascending=[True, True])
            # Display the sorted DataFrame
            st.success(f"List of gas stations ordered by (increasing) price for tank re-fill including the 2-way trip to the station")
            st.dataframe(super_filters_sorted)
            st.write("'adts / distance to reach' is the distance to the station, driving of distance, one way<br>\
                    'cost to reach' is the price of has consumption to get to the station (driving)\
                    'totalcost' is the cost for tank refill and the gas consumption for the 2-way trip to the station", unsafe_allow_html=True)
            # Save the super_filters dataframe as CSV
            # Generate a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            super_filtered_df_filename = f'super_filter_results_{timestamp}.csv'
            super_filters_sorted.to_csv(f'Data/user_specific/{super_filtered_df_filename}', index=False)
            st.success(f"Results saved to 'Data/{super_filtered_df_filename}")
            #st.write('Maps of gas stations')
            #st.components.v1.html(st.session_state.map_path)
            
            # Create and display the new map
            if not super_filters.empty:
                st.subheader("Super Recommendation Map with highlights (40 closest stations)")
                st.write("5 lowest costs stations circled in green ")
                map_with_highlights = create_map_filtered_stations_with_highlights(super_filters, user_latitude, user_longitude, selected_gas_type)
                st.components.v1.html(map_with_highlights._repr_html_(), width=1500, height=1200)
                st.subheader("Super Recommendation Map (40 closest stations)")
                super_map = create_super_filter_map(user_address, user_latitude, user_longitude, super_filters, selected_gas_type)
                folium_static(super_map, width=1500, height=1200)
            else:
                st.warning("No stations found for super recommendation.")
            
            # Save the super_filters dataframe as CSV
            super_filters.to_csv('Data/super_filters_results.csv', index=False)
            st.success("Results saved to 'Data/super_filters_results.csv'")
    else:
        st.error("Initial step 'Find Gas stations' should be processed before actioning a super recommendation")
else:
    st.write("A more precise and tailored recommendation can be made using extra information provided in left column.\
             Please press the 'super recommendation' button to obtain it once the first button and step is actioned")
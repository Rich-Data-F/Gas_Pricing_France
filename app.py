import ipaddress
from dotenv import load_dotenv
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
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text
from streamlit.runtime.scriptrunner import get_script_run_ctx
import threading
import time
from pricing_definitions import *

load_dotenv()

st.set_page_config(page_title="Gas Station and Best Price locator", page_icon="⛽", layout="wide")

def log_app_usage(conn):
    # In your main app code
    ip_address = get_remote_ip()
    print(f"Retrieved IP address: {ip_address}")
    if ip_address:
        log_connection(conn)
    else:
        print("Failed to retrieve IP address")

#def heartbeat():
#    while st.session_state.get('active',True):
#        time.sleep(30)  # Check every 30 seconds
#        update_session_duration(conn)


#if 'anonymized_ip' not in st.session_state:
#    st.session_state.anonymized_ip = ''


# Create the directory if it doesn't exist
os.makedirs('Data/usage', exist_ok=True)
# Use a relative path to the database file
db_path = os.path.join('Data', 'usage', 'usage_stats.db')
conn = st.connection('sqlite', type='sql', url=f"sqlite:///{db_path}")

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
    .stButton > Button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Use a relative path to the database file
db_path = os.path.join('Data', 'usage', 'usage_stats.db')
conn = st.connection('sqlite', type='sql', url=f"sqlite:///{db_path}")

# initialisation of session state entities
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = None
if 'user_latitude' not in st.session_state:
    st.session_state['user_latitude'] = None
if 'user_longitude' not in st.session_state:
    st.session_state['user_longitude'] = None
if 'map_path' not in st.session_state:
    st.session_state.map_path=''
if 'show_usage_stats' not in st.session_state:
    st.session_state.show_usage_stats = False
# Initialize session state
#if 'active' not in st.session_state:
#    st.session_state.active = True
#    threading.Thread(target=heartbeat).start()
# use Streamlit's run_on_save feature
if 'selected_gas_type' not in st.session_state:
    st.session_state.selected_gas_type = None
if 'radius_search' not in st.session_state:
    st.session_state.radius_search = 50  # Default value, adjust as needed
if 'anonymized_ip' not in st.session_state:
    ip_address = get_remote_ip()
    print(f"Retrieved IP address: {ip_address}")
    st.session_state.anonymized_ip = anonymize_ip(ip_address)
    print(f"Anonymized address: {st.session_state.anonymized_ip}")
    log_connection(conn)


# Initialize the database
initialize_database(conn)

# Clean up old test entries
cleanup_test_entries(conn)

# Test database connection
#test_database_insert(conn) #skipped to avoid creation of test_ip insertions

# Log the connection
ip_address = get_remote_ip()
print(f"Retrieved IP address: {ip_address}")
if ip_address:
    st.session_state.anonymized_ip = anonymize_ip(ip_address)
    log_connection(conn)
else:
    print("Failed to retrieve IP address")

# Get the API keys
try:
    openroute_api_key = get_api_key("OPENROUTE_API_KEY")
    hubspot_api_key = os.getenv("HUBSPOT_API_KEY") #get_api_key("HUBSPOT_API_KEY")
except ValueError as e:
    st.error(str(e))
    st.stop()

def main():
    st.title("Gas Station and Best Price locator")

    if 'last_activity_update' not in st.session_state:
        st.session_state.last_activity_update = datetime.now()

    # Log app usage
    log_app_usage(conn)


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
        street = st.text_input("Street Address", "rue des clefs")
    with col2:
        city = st.text_input("City", "Thônes")
    with col3:
        zipcode = st.text_input("Zipcode or at least department number", "74")

    ### verification of db connection
    # After initializing the database connection
    test_query = conn.query("SELECT COUNT(*) as count FROM usage_stats")
    print(f"Test query result: {test_query}")
    print(f"Database file path: {db_path}")
    print(f"Database file exists: {os.path.exists(db_path)}")
    
    def debug_database_content(conn):
        content = conn.query("SELECT * FROM usage_stats")
        print("Database content:")
        print(content)

    # Call this function in your main app
    debug_database_content(conn)

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
            st.session_state.gas_types = [item.split("_", 1)[-1] for item in gas_price_columns]
            print(st.session_state.gas_types)
            #default_index = gas_types.index('Gazole') if 'Gazole' in gas_types else 0
            if st.session_state.selected_gas_type:
                st.success(f"Vous avez choisi le carburant {st.session_state.selected_gas_type}")
            # Calculate distances and get closest stations
                filtered_df['distance'] = filtered_df.apply(lambda row: haversine_distance(user_latitude, user_longitude, 
                                                                                    row['latitude'], row['longitude']), axis=1)
                closest_stations = filtered_df.nsmallest(100, 'distance')
                # Create map
                m = create_map(user_address, user_latitude, user_longitude, filtered_df, st.session_state.selected_gas_type)
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
                    filtered_stations = filtered_stations[filtered_stations[f'{st.session_state.selected_gas_type}_price'].notna()]
                    # Display the filtered DataFrame
                    if not filtered_stations.empty:
                        st.dataframe(filtered_stations[['id', 'adresse', 'ville', 'distance en km', f'{st.session_state.selected_gas_type}_price', 'last_updated', "sur l'autoroute"]])
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
                (st.session_state['filtered_df']['distance']<=st.session_state.radius_search) &
                (st.session_state['filtered_df'][f'{st.session_state.selected_gas_type}_price'].notna())
                ].copy()
            user_latitude = st.session_state['user_latitude']
            user_longitude = st.session_state['user_longitude']
            if super_filters.empty:
                st.warning(f"No stations found within {st.session_state.radius_search} kms. Please try increasing the search radius.")
            else:
                #setup logging
                logging.basicConfig(level=logging.INFO)
                # ii. Calculate actual distance to station (adts) using a routing library
                def calculate_route_distance(start_coords, end_coords):
                    try:
                        # Use the API key in your Streamlit app
                        client = ors.Client(key=openroute_api_key)  # Free API key
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
                super_filters = super_filters.dropna(subset=[f'{st.session_state.selected_gas_type}_price']) # Remove rows where selected gas_type is not available
                super_filters = super_filters.dropna(subset=['adts'])  # Remove rows where route calculation failed
                # iii. Calculate costs and time to reach station
                # average price of the gas type in the 50 closest stations
                apgt50 = st.session_state['filtered_df'].nsmallest(50, 'distance')[f'{st.session_state.selected_gas_type}_price'].mean()  
                # cost to get to the station
                super_filters['cost_to_reach'] = super_filters['adts'] * st.session_state.consumption / 100 * apgt50
                super_filters['time_to_reach'] = super_filters['adts'].apply(lambda x: format_timedelta(timedelta(hours=x/50)))
                # iv. Calculate summary
                super_filters['total_cost'] = (st.session_state.tank_volume * (1 - st.session_state.tank_left)) * super_filters[f'{st.session_state.selected_gas_type}_price'] + \
                                            2*super_filters['adts'] * st.session_state.consumption / 100 * apgt50
                # Print some diagnostics
                print(f"Number of failed calculations: {(super_filters['adts'] == -1).sum()}")
                print(super_filters['adts'].describe())
    #            super_filters['adts'] = super_filters.apply(lambda row: calculate_route_distance((user_latitude, user_longitude), (row['latitude'], row['longitude'])), axis=1)
                # Display the average price of gas type in 50 closest stations
                st.write(f"Average price of {st.session_state.selected_gas_type} in 50 closest stations: {apgt50:.2f} €/L")
                # Display the super_filters dataframe
                additional_columns_to_drop=['horaires','pop']
                gas_columns_to_drop = [f"{gas}_price" for gas in st.session_state.gas_types if gas != st.session_state.selected_gas_type]
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
                    col4, col5 = st.columns(2)
                    with col4:
                        st.write("5 lowest costs stations circled in green")
                        map_with_highlights = create_map_filtered_stations_with_highlights(super_filters, user_latitude, user_longitude, st.session_state.selected_gas_type)
                        st.components.v1.html(map_with_highlights._repr_html_(), width=700, height=600)
                    with col5:
                        st.write("40 closest stations")
                        super_map = create_super_filter_map(user_address, user_latitude, user_longitude, super_filters, st.session_state.selected_gas_type)
                        folium_static(super_map, width=700, height=600)
                else:
                    st.warning("No stations found for super recommendation.")
                
                # Save the super_filters dataframe as CSV
                super_filters.to_csv('Data/super_filters_results.csv', index=False)
                st.success("Results saved to 'Data/super_filters_results.csv'")
        else:
            st.error("Initial step 'Find Gas stations' should be processed before actioning a super recommendation")
    else:
        st.write("A more precise and tailored recommendation can be made using extra information you provide in left column.\
                Please press the 'super recommendation' button to obtain it once the first 'find gas stations' button has been actioned")
        
        #sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    current_time = datetime.now()
    if (current_time - st.session_state.last_activity_update).total_seconds() > 30:
        update_last_activity(conn)
        st.session_state.last_activity_update = current_time

    # Sidebar for user inputs
    st.sidebar.title("Gas Station Finder Options")

    # 1. Gas type selection (mandatory)
    st.session_state.gas_types = ['Gazole', 'SP95', 'SP98', 'GPLc', 'E10', 'E85']  # Add all available gas types
    st.session_state.selected_gas_type = st.sidebar.selectbox("Select Gas Type", st.session_state.gas_types, index=st.session_state.gas_types.index('Gazole'))

    # Optional inputs
    st.sidebar.subheader("Optional Filters")

    # a. Outstanding autonomy
    autonomy = st.sidebar.slider("Remaining vehicule autonomy (km)", min_value=5, max_value=125, value=45, step=1)

    # b. Toilets required
    #toilets_required = st.sidebar.radio("Toilets Required", [True, False], index=0)

    # c. Spare time to save money
    #spare_time = st.sidebar.radio("Willing to spend time to save money?", ['Yes', 'No'], index=0)

    # d. Autoroutes allowed
    #autoroutes_allowed = st.sidebar.radio("Autoroutes Allowed", ["Yes", "No"], index=0)

    # e. Vehicle consumption
    st.session_state.consumption = st.sidebar.number_input("Vehicle Consumption (L/100km)", value=6.0, min_value=0.5, step=0.5)

    # f. Gas tank total volume
    st.session_state.tank_volume = st.sidebar.slider("Gas Tank Total Volume (L)", min_value=5, max_value=80, value=45, step=1)

    # g. Current gas tank volume left (fraction)
    st.session_state.tank_left = st.sidebar.slider("Current Gas Tank Volume Left (fraction)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

    # h. radius search size in kms
    st.session_state.radius_search = st.sidebar.slider("Size of radius search (km)", min_value=1, max_value=(int(autonomy)-5), value=50, step=2)

    st.sidebar.write("The recommendation is limited to the 40 closest stations to the selected address, which may prevail on the above-defined distance and criteria")

    # Display the visitor count
    st.sidebar.write(f"This application has been used {get_unique_users(conn)} times.")
    
    # Optional: Display the extra stats
    extra_stats=st.sidebar.toggle("App usage extra statistics")
    if extra_stats:
        display_usage_stats(conn)

    st.sidebar.write("This app allows you to submit customer feedback directly to our HubSpot CRM.")
    with st.sidebar.expander("Submit Feedback"):
        st.title("Submit a Ticket")
        subject = st.text_input("Subject")
        description = st.text_area("Description")
        # Define category options with user-friendly labels
        category_options = {
            "Product Issue": "PRODUCT_ISSUE",
            "Billing Issue": "BILLING_ISSUE",
            "Feature Request": "FEATURE_REQUEST",
            "General Inquiry": "GENERAL_INQUIRY"
        }
        selected_category = st.selectbox(
            "Category",
            options=list(category_options.keys()),
            index=None,
            format_func=lambda x: x,
            placeholder="Select a category..."
        )
        # Define priority options with user-friendly labels
        priority_options = {
            "Low": "LOW",
            "Medium": "MEDIUM",
            "High": "HIGH"
        }
        selected_priority = st.selectbox(
            "Priority (optional)",
            options=[None] + list(priority_options.keys()),
            index=0,
            format_func=lambda x: "Select priority..." if x is None else x
        )
        if st.button("Submit Ticket"):
            if not subject or not description or not selected_category:
                st.error("Please fill in all required fields.")
            else:
                # Map the selected options to their API values
                category = category_options[selected_category]
                priority = priority_options[selected_priority] if selected_priority else None            
                success, message = submit_ticket(category, subject, description, priority)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    if st.toggle("I'm done"):
        st.session_state.active = False
        update_session_duration(conn)
        #if duration is not None:
        #    st.write(f"App usage duration: {duration:.2f} seconds")
        #else:
        #    st.write("App usage duration: Not available")
        st.write("Thank you for using the app. You can now close this tab.")
        st.stop()


if __name__ == "__main__":
    main()
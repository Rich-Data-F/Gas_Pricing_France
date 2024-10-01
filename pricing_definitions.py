import ipaddress
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import locale
import pandas as pd
import duckdb
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from dotenv import load_dotenv
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
import numpy as np
import os
from cryptography.fernet import Fernet
import requests
import json
import logging
from sqlalchemy import create_engine, text
import time


#### definition of functions for logging connections and some stats

def get_remote_ip():
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None
        return ctx.session_info.request.remote_ip
    except Exception as e:
        print(f"Error getting remote IP: {str(e)}")
        return None

def anonymize_ip(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.version == 4:
            return str(ipaddress.ip_network(f"{ip}/24", strict=False).network_address)
        elif ip_obj.version == 6:
            return str(ipaddress.ip_network(f"{ip}/48", strict=False).network_address)
    except ValueError:
        return None  # Invalid IP address

def initialize_database(conn):
    with conn.session as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                connection_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                anonymized_ip TEXT,
                last_activity_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_duration INTEGER DEFAULT 0
            )
        """))
        session.commit()

def log_connection(conn):
    ip_address = get_remote_ip()
    print(f"Attempting to log connection for IP: {ip_address}")
    if ip_address is None or ip_address == "127.0.0.1":
        ip_address = "unknown"
#    st.session_state.anonymized_ip = anonymize_ip(ip_address)
    print(f"Anonymized IP: {st.session_state.anonymized_ip}")
    if st.session_state.anonymized_ip:
        current_time = datetime.now()
        try:
            with conn.session as session:
                # Create new session
                session.execute(text("""
                    INSERT INTO usage_stats (anonymized_ip, connection_time, last_activity_time) 
                    VALUES (:ip, :current_time, :current_time)
                """), {"ip": st.session_state.anonymized_ip, "current_time": current_time})
                session.commit()
                print("Session committed successfully")
        except Exception as e:
            print(f"Error in log_connection: {str(e)}")
    else:
        print("Failed to anonymize IP")

def get_duration_of_exposition(conn):
    try:
        with conn.session as session:
            result = session.execute(text("""
                SELECT connection_time, last_activity_time
                FROM usage_stats
                WHERE anonymized_ip = :ip
                ORDER BY connection_time DESC
                LIMIT 1
            """), {"ip": st.session_state.anonymized_ip}).fetchone()
            
            if result:
                connection_time, last_activity_time = result
                duration = last_activity_time - connection_time
                return duration.total_seconds()
            else:
                return None
    except Exception as e:
        print(f"Error in get_duration_of_exposition: {str(e)}")
        return None

def get_average_session_duration(conn):
    return conn.query("""
        SELECT AVG(session_duration) as avg_duration 
        FROM usage_stats
    """).iloc[0]['avg_duration']

def get_total_session_duration(conn):
    return conn.query("""
        SELECT SUM(session_duration) as total_duration 
        FROM usage_stats
    """).iloc[0]['total_duration']

def get_longest_session(conn):
    return conn.query("""
        SELECT MAX(session_duration) as max_duration 
        FROM usage_stats
    """).iloc[0]['max_duration']

def get_total_connections(conn):
    return conn.query("SELECT COUNT(*) as total FROM usage_stats").iloc[0]['total']

def get_unique_users(conn):
#    return conn.query("SELECT COUNT(DISTINCT anonymized_ip) as unique_users FROM usage_stats").iloc[0]['unique_users']
    result = conn.query("SELECT COUNT(DISTINCT anonymized_ip) as unique_users FROM usage_stats").iloc[0]['unique_users']
    print(f"Unique users query result: {result}")
    return result if result is not None else 0

def display_usage_stats(conn):
    total_connections = get_total_connections(conn)
    unique_users = get_unique_users(conn)
    avg_duration = get_average_session_duration(conn)
    total_duration = get_total_session_duration(conn)
    st.sidebar.write(f"Total connections: {total_connections}")
    st.sidebar.write(f"Unique users: {unique_users}")
    if avg_duration is not None:
        st.sidebar.write(f"Average session duration: {avg_duration:.2f} seconds")
    else:
        st.sidebar.write("Average session duration: N/A")
    st.sidebar.write(f"Total session duration: {total_duration} seconds")

######## end of logging tracking

def decrypt_env():
    with open("secret.key", "rb") as key_file:
        key = key_file.read()
    fernet = Fernet(key)
    with open('.env.encrypted', 'rb') as enc_file:
        encrypted = enc_file.read()
    decrypted = fernet.decrypt(encrypted)
    return decrypted.decode()

def get_api_key(key_name):
    print(f"Attempting to get API key for: {key_name}")
    
    # Try Streamlit secrets
    try:
        api_key = st.secrets[key_name]
        if api_key:
            print(f"API key found in Streamlit secrets.")
            return api_key
    except (FileNotFoundError, KeyError):
        print(f"Key not found in Streamlit secrets, trying decrypted .env")
    
    # Try decrypted .env file
    try:
        decrypted_env = decrypt_env()  # Ensure your decrypt_env function is defined
        for line in decrypted_env.split('\n'):
            if line.startswith(f'{key_name}='):
                api_key = line.split('=')[1].strip()
                if api_key:
                    print(f"API key found in decrypted .env.")
                    return api_key
    except Exception as e:
        print(f"Error decrypting .env file: {e}")
    
    # If we've reached this point, both Streamlit secrets and decrypted .env have failed
    print(f"Key not found in Streamlit secrets or decrypted .env, trying regular .env")
    
    # Only load from regular .env if previous methods failed
    load_dotenv()
    api_key = os.getenv(key_name)
    
    if api_key:
        print(f"API key found in regular .env.")
        return api_key
    else:
        print(f"API key not found in any source.")
        return None


# HubSpot API configuration
BASE_URL = "https://api.hubapi.com"
#ACCESS_TOKEN = os.getenv("HUBSPOT_API_KEY")

def submit_ticket(category, subject, description, priority=None):
    url = "https://api.hubapi.com/crm/v3/objects/tickets"
    
    ACCESS_TOKEN = os.environ.get('HUBSPOT_API_KEY')
    if not ACCESS_TOKEN:
        logging.error("ACCESS_TOKEN is not set or empty")
        return False, "Error: ACCESS_TOKEN is not set or empty"

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "properties": {
            "subject": subject,
            "content": description,
            "hs_pipeline": "0",
            "hs_pipeline_stage": "864905971",
            "hs_ticket_category": category
        }
    }
    
    if priority:
        data["properties"]["hs_ticket_priority"] = priority

    try:
        logging.info(f"Submitting ticket: {data}")
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        logging.info(f"Response status: {response.status_code}")
        logging.info(f"Response content: {response.text}")
        return True, "Ticket submitted successfully"
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error: {e}")
        return False, f"Error submitting ticket: {str(e)}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error submitting ticket: {str(e)}")
        return False, f"Error submitting ticket: {str(e)}"

def submit_feedback(category, subject, body, rating):
    url = "https://api.hubapi.com/crm/v3/objects/feedback_submissions"
    headers = {
        "Authorization": f"Bearer {YOUR_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "properties": {
            "hs_timestamp": datetime.utcnow().isoformat() + "Z",
            "hs_feedback_source": "Streamlit App",
            "hs_feedback_subject": f"{category}: {subject}",
            "hs_feedback_body": body,
#            "feedback_category": category,
            "hs_feedback_rating": str(rating)
        }
    }
    response = requests.post(url, json=data, headers=headers)
    return response.status_code == 200

def get_hubspot_properties():
    """Retrieve properties for feedback submissions from HubSpot."""
    url = f"{BASE_URL}/crm/v3/properties/feedback_submissions/batch/create"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        properties = response.json()
        return True, properties
    except requests.exceptions.RequestException as e:
        return False, f"Error retrieving properties: {str(e)}"

def submit_feedback(category, subject, body, rating):
    """Submit feedback to HubSpot."""
    url = f"{BASE_URL}/crm/v3/objects/feedback_submissions/batch/create"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": [
            {
                "properties": {
                    "hs_timestamp": datetime.utcnow().isoformat() + "Z",
                    "hs_feedback_source": "Streamlit App",
                    "hs_feedback_subject": f"{category}: {subject}",
                    "hs_feedback_body": body,
                    "hs_feedback_rating": str(rating)
                }
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return True, "Feedback submitted successfully!"
    except requests.exceptions.RequestException as e:
        return False, f"Error submitting feedback: {str(e)}"


def get_api_key(key_name):
    # First, try to get the API key from Streamlit secrets
    try:
        return st.secrets[key_name]
    except (FileNotFoundError, KeyError):
        # If not found in secrets, try to load from decrypted .env file
        try:
            decrypted_env = decrypt_env()
            for line in decrypted_env.split('\n'):
                if line.startswith(f'{key_name}='):
                    return line.split('=')[1].strip()
        except Exception as e:
            print(f"Error decrypting .env file: {e}")
        # If still not found, try loading from regular .env file
        load_dotenv()
        api_key = os.getenv(key_name)
        if api_key:
            return api_key        
        # If API key is not found anywhere, raise an error
        raise ValueError(f"{key_name} not found in Streamlit secrets, decrypted .env file, or regular .env file")

def get_file_info(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    # Get file creation time (upload time)
    creation_time = os.path.getctime(file_path)
    creation_datetime = datetime.fromtimestamp(creation_time)
    print(creation_datetime)
    # Get file modification time
    modification_time = os.path.getmtime(file_path)
    modification_datetime = datetime.fromtimestamp(modification_time)
    print(modification_datetime)
    return {
        "file_name": file_name,
        "creation_date": creation_datetime.strftime("%Y-%m-%d"),
        "creation_time": creation_datetime.strftime("%H:%M:%S"),
        "modification_date": modification_datetime.strftime("%Y-%m-%d"),
        "modification_time": modification_datetime.strftime("%H:%M:%S"),
    }

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for pdv in root.findall('pdv'):
        row = pdv.attrib.copy()
        for child in pdv:
            if child.tag == 'horaires':
                row['automate-24-24'] = child.get('automate-24-24')
                horaires = {}
                for jour in child.findall('jour'):
                    jour_info = jour.attrib.copy()
                    horaire = jour.find('horaire')
                    if horaire is not None:
                        jour_info.update(horaire.attrib)
                    horaires[jour.get('nom')] = jour_info
                row['horaires'] = horaires
            elif child.tag == 'services':
                row['services'] = [service.text for service in child.findall('service')]
            elif child.tag == 'prix':
                prix_info = child.attrib.copy()
                row[f"prix_{prix_info['nom']}"] = prix_info
            elif child.tag == 'rupture':
                rupture_info = child.attrib.copy()
                row[f"rupture_{rupture_info['nom']}"] = rupture_info
            else:
                row[child.tag] = child.text
        data.append(row)
    return pd.DataFrame(data)

def process_gas_prices(df):
    # Function to extract date from price dictionary
    def extract_date(price_dict):
        return pd.to_datetime(price_dict['maj']) if isinstance(price_dict, dict) and 'maj' in price_dict else pd.NaT
    # Function to extract price from price dictionary
    def extract_price(price_dict):
        return float(price_dict['valeur']) if isinstance(price_dict, dict) and 'valeur' in price_dict else None
    # Identify gas price columns
    gas_price_columns = [col for col in df.columns if col.startswith('prix_')]
    # Process each gas price column
    for col in gas_price_columns:
        fuel_type = col.split('_')[1]  # Extract fuel type from column name
        df[f'{fuel_type}_price'] = df[col].apply(extract_price)
        df[f'{col}_date'] = df[col].apply(extract_date)
    # Find the latest date for each row
    date_columns = [col for col in df.columns if col.endswith('_date')]
    df['last_updated'] = df[date_columns].max(axis=1)
    # Convert 'last_updated' to date (removing time information)
    df['last_updated'] = df['last_updated'].dt.date
    # Drop the original price columns and temporary date columns
    columns_to_drop = gas_price_columns + date_columns
    df = df.drop(columns=columns_to_drop)
    # Sort the DataFrame by 'last_updated' in descending order
    df = df.sort_values('last_updated', ascending=False)
    return df

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

def get_lat_long(address):
    default_lat, default_lon = 48.856614, 2.352222  # Hotel de Ville, Paris
    geolocator = Nominatim(user_agent="my_app")
    max_attempts = 3
    for attempt in range(max_attempts):
        try: 
            location = geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
            else:
                print(f"{address} could not be found: ")
                if attempt < max_attempts - 1:
                    prompt_for_address()
                else:
                    raise GeocodingError("Unable to geocode the address after multiple attempts")
        except (GeocoderServiceError, GeocoderTimedOut) as e:
            print(f"Error occurred: {e}")
            if attempt < max_attempts - 1:
                print("Retrying...")
            else:
                raise GeocodingError(f"Geocoding failed after {max_attempts} attempts") from e
    raise GeocodingError("Unexpected error in geocoding")

# collection of the origin address from the user
def prompt_for_address():
    while True:
        user_street = input("Enter street from the origin address: ").strip()
        if user_street:
            break
        print("Street address cannot be empty. Please try again.")
    while True:
        user_city = input("Enter city: ").strip()
        if user_city:
            break
        print("City cannot be empty. Please try again.")
    while True:
        user_postal_code = input("Enter postal code or at least the department number (press Enter to skip): ").strip()
        if user_postal_code:
            break
        print("Postal code cannot be empty. Please try again.")
    while True:
        user_country = input("Enter country (just press Enter for France set as Default): ").strip()
        if not user_country:
            user_country = "France"
        break
    user_full_address = f"{user_street}, {user_postal_code}, {user_city}, {user_country}"
    return user_full_address, user_postal_code

# collection of the required gas type from the existing list
def select_gas_type(gas_types):
    for index, gas_type in enumerate(gas_types, start=1):
        print(f"{index}. {gas_type}")
    while True:
        try:
            selection = int(input("\nEnter the number of your chosen gas type: "))
            if 1 <= selection <= len(gas_types):
                return gas_types[selection - 1]
            else:
                print("Invalid selection. Please choose a number from the list.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def read_adjacent_departments(file_path):
    adjacent_deps = {}
    with open(file_path, 'r') as file:
        for line in file:
            department, adjacents = line.strip().split(':')
            adjacent_deps[department] = adjacents.split(',')
        print(adjacent_deps)#debugging
    return adjacent_deps

def get_relevant_departments(user_postal_code, adjacent_deps):
    user_department = user_postal_code[:2]
    relevant_depts = [user_department] + adjacent_deps.get(user_department, [])
    print(relevant_depts)#debugging
    return relevant_depts
# department at least requested for postal code to user

def filter_gas_stations_df(df, relevant_depts):
    filtered_df=df[df['department'].isin(relevant_depts)]
    filtered_df.head() #debugging
    return filtered_df

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}"

def create_map(user_address, user_latitude, user_longitude, filtered_df, selected_gas_type):
    m = folium.Map(location=[user_latitude, user_longitude], zoom_start=11)  
    folium.Marker(
        [user_latitude, user_longitude],
        popup=f"Your Location:<br>{user_address}",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)
    for _, station in filtered_df.iterrows():
        # Use .get() method to avoid KeyError
        name = station.get('id', station.get('id', 'Unknown Station'))
        ville = station.get('ville','N/A')
        adresse = station.get('adresse','N/A')
        price = station.get(f'{selected_gas_type}_price', 'N/A')
        last_update = station.get('last_updated', 'Unknown')
        popup_html = f"""
        <b>{name}</b><br>
        {selected_gas_type}: {price} €<br>
        Last updated: {last_update}
        """
        
        # Create a label for the marker
        label = f"adresse: {adresse}, ville: {ville}<br>Price: f{price:.2f} €"
    
        # Use .get() method for latitude and longitude as well
        lat = station.get('latitude', station.get('lat'))
        lon = station.get('longitude', station.get('lon'))
        if lat is not None and lon is not None:
            folium.Marker(
                [lat, lon],
                popup=popup_html, 
                tooltip=label,
                icon=folium.Icon(color='blue', icon='gas-pump', prefix='fa')
            ).add_to(m)            
    return m

def create_super_filter_map(user_address, user_latitude, user_longitude, super_filters, selected_gas_type):
    m = folium.Map(location=[user_latitude, user_longitude], zoom_start=10)
    
    # Add marker for user location
    folium.Marker(
        [user_latitude, user_longitude],
        popup=user_address,
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)

    # Add markers for gas stations
    for _, row in super_filters.iterrows(): 
        # Determine the relevant price based on the selected gas type
        relevant_price = row[f'{selected_gas_type}_price']
        
        # Create a label for the marker
        label = f"{row['adresse']}, {row['ville']}<br>Price: {relevant_price:.2f} €<br>Total Cost: {row['total_cost']:.2f} €"
        
        # Create a popup with detailed information
        popup_html = f"""
        <b>{row['adresse']}, {row['ville']}</b><br>
        Price: {relevant_price:.2f} €<br>
        Distance: {row['distance']:.2f} km<br>
        Route Distance: {row['adts']:.2f} km<br>
        Cost to Reach: {row['cost_to_reach']:.2f} €<br>
        Time to Reach: {row['time_to_reach']}<br>
        Total Cost: {row['total_cost']:.2f} €
        """
        
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=label,  # This will show up on hover
            icon=folium.Icon(color='green', icon='gas-pump', prefix='fa')
        ).add_to(m)
    return m

def create_map_filtered_stations_with_highlights(super_filters, user_latitude, user_longitude, selected_gas_type):
    # Sort the DataFrame by total_cost and select the top 5
    top_5_stations = super_filters.sort_values('total_cost').head(5)

    # Create a base map
    m = folium.Map(location=[user_latitude, user_longitude], zoom_start=12)

    # Add a marker for the user's location
    folium.Marker(
        [user_latitude, user_longitude],
        popup="Your Location",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # Create a MarkerCluster for all stations
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers for all stations
    for idx, row in super_filters.iterrows():
        # Prepare popup content
        popup_content = f"""
        <b>ID:</b> {row['id']}<br>
        <b>Address:</b> {row['adresse']}<br>
        <b>City:</b> {row['ville']}<br>
        <b>Postal Code:</b> {row['cp']}<br>
        <b>{selected_gas_type} Price:</b> {row[f'{selected_gas_type}_price']}<br>
        <b>Last Updated:</b> {row['last_updated']}<br>
        <b>Actual Distance:</b> {row['adts']:.2f} km<br>
        <b>Total Cost for Refill (2-way trip):</b> {row['total_cost']:.2f} €
        """

        # Prepare tooltip content
        tooltip_content = f"{row['adresse']}, {row['ville']}<br>{selected_gas_type} Price: {row[f'{selected_gas_type}_price']}<br>Actual Distance: {row['adts']:.2f} km<br>Total Cost: {row['total_cost']:.2f} €"

        # Add marker to cluster
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=tooltip_content,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

    # Add special markers for the top 5 stations
    for idx, row in top_5_stations.iterrows():
        popup_content = f"""
        <b>ID:</b> {row['id']}<br>
        <b>Address:</b> {row['adresse']}<br>
        <b>City:</b> {row['ville']}<br>
        <b>Postal Code:</b> {row['cp']}<br>
        <b>{selected_gas_type} Price:</b> {row[f'{selected_gas_type}_price']}<br>
        <b>Last Updated:</b> {row['last_updated']}<br>
        <b>Actual Distance:</b> {row['adts']:.2f} km<br>
        <b>Total Cost for Refill (2-way trip):</b> {row['total_cost']:.2f} €
        """

        tooltip_content = f"{row['adresse']}, {row['ville']}<br>{selected_gas_type} Price: {row[f'{selected_gas_type}_price']}<br>Actual Distance: {row['adts']:.2f} km<br>Total Cost: {row['total_cost']:.2f} €"

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=tooltip_content,
            icon=folium.Icon(color='green', icon='star')
        ).add_to(m)

        # Add a circle to highlight the area
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=500,  # 500 meters radius, adjust as needed
            color='green',
            fill=True,
            fillColor='green',
            fillOpacity=0.2
        ).add_to(m)

    return m

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    # Convert latitude and longitude to floats and then radians
    lat1, lon1, lat2, lon2 = map(lambda x: np.radians(float(x)),[lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance
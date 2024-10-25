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
from streamlit.web.server.websocket_headers import  *
import bcrypt
from functools import wraps
import sqlite3
import hashlib
import pathlib
from pathlib import Path
from bs4 import BeautifulSoup
import shutil

# User roles
USER_ROLE = 'user'
ADMIN_ROLE = 'admin'

# Initialize SQLite database

def inject_ga():
    GA_ID="google_analytics"
    GA_JS= """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-Z0YKHBLBKY"></script> 
    <script> 
        window.dataLayer = window.dataLayer || []; 
        function gtag(){dataLayer.push(arguments);} 
        gtag('js', new Date());
        gtag('config', 'G-Z0YKHBLBKY');
        </script> 
        """
    # Insert the script in the head tag of the static template inside your virtual
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    logging.info(f'editing {index_path}')
#    index_path = Path('/path/to/your/index.html')  # Update this to your actual index path
    bck_index = Path('Data')  # Update this to a writable location
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")
    if not soup.find(id=GA_ID): 
        bck_index = index_path.with_suffix('.bck')
        if bck_index.exists():
            shutil.copy(bck_index, index_path)  
        else:
            shutil.copy(index_path, bck_index)  
        html = str(soup)
        new_html = html.replace('<head>', '<head>\n' + GA_JS)
        index_path.write_text(new_html)


def init_users_db():
    # Create the Data directory if it doesn't exist
    os.makedirs('Data', exist_ok=True)
    # Check if the database file exists
    if not os.path.exists('Data/users.db'):
        conn = sqlite3.connect(os.path.join('Data','users.db'))
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS connection_logs
                    (username TEXT, login_time TIMESTAMP, logout_time TIMESTAMP)''')
        # Add default admin user if not exists
        c.execute("SELECT * FROM users WHERE username = 'rich'")
        if c.fetchone() is None:
            hashed_password = bcrypt.hashpw('adminpass'.encode('utf-8'), bcrypt.gensalt())
            c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                      ('rich', hashed_password, ADMIN_ROLE))  # Ensure Rich is assigned ADMIN_ROLE
        conn.commit()
        conn.close()
    else:
        print("Database already exists.")

def auth_block():
    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None
    if st.session_state.user is None:
        st.sidebar.title("Authentication")
        tab1, tab2, tab3 = st.sidebar.tabs(["Login", "Sign Up", "Reset Password"])
        with tab1:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if login(login_username, login_password):
                    st.success(f"Logged in as {login_username}")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            new_username = st.text_input("New Username", key="new_username")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            if st.button("Sign Up"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif signup(new_username, new_password):
                    st.success("Account created successfully. Please log in.")
                else:
                    st.error("Username already exists or an error occurred")
        
        with tab3:
            reset_username = st.text_input("Username for Password Reset", key="reset_username")
            new_password = st.text_input("New Password", type="password", key="reset_new_password")
            confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_password")
            if st.button("Reset Password"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif reset_password(reset_username, new_password):
                    st.success("Password reset successfully. Please log in.")
                else:
                    st.error("Username not found or an error occurred")
    else:
        st.sidebar.write(f"Logged in as: {st.session_state.user}")
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()

# Session management
if 'user' not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.login_time = None
# Initialize session state for form data
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None
if 'new_role' not in st.session_state:
    st.session_state.new_role = None

def login(username, password):
    conn = sqlite3.connect(os.path.join('Data', 'users.db'))
    c = conn.cursor()
    c.execute("SELECT password, role FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[0]):
        st.session_state.user = username
        st.session_state.role = user[1]
        return True
    return False

def signup(username, password):
    conn = sqlite3.connect(os.path.join('Data', 'users.db'))
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return False
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
              (username, hashed_password, 'user'))
    conn.commit()
    conn.close()
    return True

def reset_password(username, new_password):
    conn = sqlite3.connect(os.path.join('Data', 'users.db'))
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    if c.fetchone() is None:
        conn.close()
        return False
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    c.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_password, username))
    conn.commit()
    conn.close()
    return True

def fetch_users():
    # Simulate fetching users from a database
    conn = sqlite3.connect(os.path.join('Data', 'users.db'))
    c = conn.cursor()
    c.execute("SELECT username FROM users")
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return users

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if st.session_state.user is None:
            st.warning("Please log in to access this feature.")
            return None
        return func(*args, **kwargs)
    return wrapper

def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if st.session_state.role != ADMIN_ROLE:
            st.warning("Admin access required for this feature.")
            return None
        return func(*args, **kwargs)
    return wrapper

def logout():
    if st.session_state.user:
        conn = sqlite3.connect(os.path.join('Data', 'users.db'))
        c = conn.cursor()
        logout_time = datetime.now()
        c.execute("UPDATE connection_logs SET logout_time = ? WHERE username = ? AND login_time = ?",
                  (logout_time, st.session_state.user, st.session_state.login_time))
        conn.commit()
        conn.close()
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.login_time = None
    st.sidebar.success("Logged out successfully")

def print_users():
    conn = sqlite3.connect(os.path.join('Data', 'users.db'))
    c = conn.cursor()
    c.execute("SELECT username, role FROM users")
    users = c.fetchall()
    conn.close()
    for user in users:
        print(f"Username: {user[0]}, Role: {user[1]}")


@admin_required
def change_user_role():
    st.title("Change User Role")
    # Fetch users from the database
    users = fetch_users()
    roles = ['user', 'admin']
    # Use session state to store the selected user and role
    selected_user = st.selectbox(
        "Select User", 
        users, 
        index=users.index(st.session_state.selected_user) if st.session_state.selected_user in users else 0,
        key='selected_user'
    )
    new_role = st.selectbox(
        "Select New Role", 
        roles, 
        index=roles.index(st.session_state.new_role) if st.session_state.new_role in roles else 0,
        key='new_role'
    )
    if st.button("Change Role"):
        # Simulate updating the user's role in the database
        conn = sqlite3.connect(os.path.join('Data', 'users.db'))
        c = conn.cursor()
        c.execute("UPDATE users SET role = ? WHERE username = ?", (new_role, selected_user))
        conn.commit()
        conn.close()
        st.success(f"Changed {selected_user}'s role to {new_role}")
    
# Super Recommendation feature accessible only to admin users
@login_required
def show_super_recommendation():
    if st.button("Super Recommendation"):
        st.toast("Processing calculations... Should take less than a minute ...")
        if st.session_state['filtered_df'] is not None and st.session_state['user_latitude'] is not None and st.session_state['user_longitude'] is not None:
            # Perform the advanced filtering and calculations
            super_filters = st.session_state['filtered_df'][
                (st.session_state['filtered_df']['distance']<=st.session_state.radius_search) &
                (st.session_state['filtered_df'][f'{st.session_state.selected_gas_type}_price'].notna())
            ].copy()
            user_latitude = st.session_state['user_latitude']
            user_longitude = st.session_state['user_longitude']
            if super_filters.empty:
                st.warning(f"No stations found within {st.session_state.radius_search} kms. Please try increasing the search radius.")
            else:
                # Setup logging
                logging.basicConfig(level=logging.INFO)
                # Calculate actual distance to station (adts) using a routing library
                def calculate_route_distance(start_coords, end_coords):
                    try:
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
                # Fill the remaining rows with NaN
                super_filters['adts'] = super_filters['adts'].fillna(np.nan)
                
                super_filters = super_filters.dropna(subset=[f'{st.session_state.selected_gas_type}_price']) # Remove rows where selected gas_type is not available
                super_filters = super_filters.dropna(subset=['adts'])  # Remove rows where route calculation failed
                
                # Calculate costs and time to reach station
                apgt50 = st.session_state['filtered_df'].nsmallest(50, 'distance')[f'{st.session_state.selected_gas_type}_price'].mean()  
                super_filters['cost_to_reach'] = super_filters['adts'] * st.session_state.consumption / 100 * apgt50
                super_filters['time_to_reach'] = super_filters['adts'].apply(lambda x: format_timedelta(timedelta(hours=x/50)))
                
                # Calculate summary
                super_filters['total_cost'] = (st.session_state.tank_volume * (1 - st.session_state.tank_left)) * super_filters[f'{st.session_state.selected_gas_type}_price'] + \
                                            2*super_filters['adts'] * st.session_state.consumption / 100 * apgt50
                
                # Print some diagnostics
                print(f"Number of failed calculations: {(super_filters['adts'] == -1).sum()}")
                print(super_filters['adts'].describe())
                
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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                super_filtered_df_filename = f'super_filter_results_{timestamp}.csv'
                super_filters_sorted.to_csv(f'Data/user_specific/{super_filtered_df_filename}', index=False)
                st.success(f"Results saved to 'Data/{super_filtered_df_filename}")
                
                # Create and display the new map
                if not super_filters.empty:
                    col4, col5 = st.columns(2)
                    with col4:
                        st.write("5 lowest costs stations circled in green")
                        map_with_highlights = create_map_filtered_stations_with_highlights(super_filters, user_latitude, user_longitude, st.session_state.selected_gas_type)
                        st.components.v1.html(map_with_highlights._repr_html_(), width=700, height=600)
                    with col5:
                        st.write("40 closest stations")
                        super_map = create_super_filter_map(st.session_state.user_address, user_latitude, user_longitude, super_filters, st.session_state.selected_gas_type)
                        folium_static(super_map, width=700, height=600)
                else:
                    st.warning("No stations found for super recommendation.")
                
                # Save the super_filters dataframe as CSV
                super_filters.to_csv('Data/super_filters_results.csv', index=False)
                st.success("Results saved to 'Data/super_filters_results.csv'")
        else:
            st.error("Initial step 'Find Gas stations' should be processed before actioning a super recommendation")

load_dotenv()

st.set_page_config(page_title="Gas Station and Best Price locator", page_icon="⛽", layout="wide")

def log_app_usage(conn):
    ip_address = get_remote_ip()
    print(f"Retrieved IP address from log_app_usage / get remote: {ip_address}")
    if ip_address:
        log_connection(conn)
    else:
        print("Failed to retrieve IP address from log_app_usage / get remote")

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

# Initialization of session state entities
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
if 'active' not in st.session_state:
    st.session_state.active = True
#    threading.Thread(target=heartbeat).start()
# use Streamlit's run_on_save feature
if 'selected_gas_type' not in st.session_state:
    st.session_state.selected_gas_type = None
if 'radius_search' not in st.session_state:
    st.session_state.radius_search = 50  # Default value, adjust as needed
if 'anonymized_ip' not in st.session_state:
    ip_address = get_remote_ip()
    print(f"Retrieved IP address from get_remote_ip: {ip_address}")
    st.session_state.anonymized_ip = anonymize_ip(ip_address)
    print(f"Anonymized address from anonymize_ip fcn: {st.session_state.anonymized_ip}")
    log_connection(conn)

# Initialize the database
initialize_database(conn)

# Clean up old test entries
#cleanup_test_entries(conn)

# Log the connection
ip_address = get_remote_ip()
if ip_address:
    st.session_state.anonymized_ip = anonymize_ip(ip_address)
    log_connection(conn)
else:
    print("Failed to retrieve IP address from get_forwarded_ip")

# Get the API keys
try:
    openroute_api_key = get_api_key("OPENROUTE_API_KEY")
    hubspot_api_key = os.getenv("HUBSPOT_API_KEY")
except ValueError as e:
    st.error(str(e))
    st.stop()

def main():
    GA_TRACKING_ID = 'G-Z0YKHBLBKY'
    # Inject Google Analytics tracking code into the Streamlit app
    st.markdown(f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{GA_TRACKING_ID}');
        </script>
    """, unsafe_allow_html=True)
#    inject_ga()
    st.title("Gas Station and Best Price locator")
    # Initialize the database
    init_users_db()
    print_users()
    # Authentication block
    auth_block()
    # Rest of your main function...
    if st.session_state.user:
        st.write(f"Welcome, {st.session_state.user}!")
        if st.session_state.role == ADMIN_ROLE:
            if st.button("Change User Role"):
                change_user_role()
    # Log app usage
    log_app_usage(conn)
    folder_path='./Data/'
    file_name='PrixCarburants_instantane.xml'
    file_info = get_file_info(folder_path, file_name)
    update_datetime = datetime.strptime(f"{file_info['creation_date']} {file_info['creation_time']}", "%Y-%m-%d %H:%M:%S")
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
        street = st.text_input("Street Address", "rue des clefs",key='street')
    with col2:
        city = st.text_input("City", "Thônes",key='city')
    with col3:
        zipcode = st.text_input("Zipcode or at least department number", "74",key='zipcode')

    country = "France"
    st.session_state.user_address = f"{street}, {zipcode}, {city}, {country}"

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
    
    if st.button("Find Gas Stations"):
        result = get_coordinates(st.session_state.user_address)
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
                m = create_map(st.session_state.user_address, user_latitude, user_longitude, filtered_df, st.session_state.selected_gas_type)
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
            st.write("A more precise and tailored recommendation can be made using extra information provided in left column. The 'Super Recommendation' button can be actioned once the first 'Find Gas Stations' step is actioned")
        else:
            st.error("Could not find coordinates for the given address")

    show_super_recommendation()

    current_time = datetime.now()
#    if (current_time - st.session_state.last_activity_update).total_seconds() > 30:
#        update_last_activity(conn)
#        st.session_state.last_activity_update = current_time

    # Optional: Display the extra stats for admin users
    if st.session_state.role == ADMIN_ROLE:
        extra_stats = st.sidebar.toggle("App usage extra statistics")
        if extra_stats:
            display_usage_stats(conn)

    if st.toggle("I'm done"):
        st.session_state.active = False
        update_session_duration(conn)
        st.write("Thank you for using the app. You can now close this tab.")
        st.stop()

if __name__ == "__main__":
    main()
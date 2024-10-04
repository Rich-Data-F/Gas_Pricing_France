import pandas as pd
import requests
import xmltodict
import json
import zipfile
import io
import os
from pricing_definitions import get_file_info, parse_xml
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google.cloud import bigquery
import xml.etree.ElementTree as ET

# Download XML from website
# Step 1: Download the ZIP file from the HTML address
url = "https://donnees.roulez-eco.fr/opendata/instantane"
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
# Step 2: Extract the XML file from the ZIP
xml_content = zip_file.read('PrixCarburants_instantane.xml')
# Step 3: Save the XML content to the Data folder
data_folder = "Data"  # Adjust this path as needed
file_path = os.path.join(data_folder, "PrixCarburants_instantane_forBigQuery.xml")
with open(file_path, "wb") as f:
    f.write(xml_content)
# Step 4: Read the XML content using pd.read_xml()
df = pd.read_xml(file_path, xpath='.//pdv',
                encoding='ISO-8859-1',
                dtype=str)
# Step 5: display the refreshed date
folder_path='./Data/'
file_name='PrixCarburants_instantane_forBigQuery.xml'
file_info = get_file_info(folder_path, file_name)
# Parse the date and time strings back into a datetime object
update_datetime = datetime.strptime(f"{file_info['creation_date']} {file_info['creation_time']}", "%Y-%m-%d %H:%M:%S")
# Format it as desired
formatted_datetime = update_datetime.strftime("%d %B %Y à %Hh%M")
print(f"Price information have been last updated on {formatted_datetime}.")
file_info = get_file_info(folder_path, file_name)


# Load environment variables from .env file
load_dotenv()

# Now you can access the environment variable
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

project_id=os.getenv('PROJECT_ID')
dataset_id=os.getenv('DATASET_ID')
table_id=f"gas_pricing_from_opendata_{datetime.today().strftime('%Y_%m_%d')}"
# Use the credentials in your BigQuery client initialization
client = bigquery.Client(project=os.getenv('PROJECT_ID'))
job_config = bigquery.LoadJobConfig()
job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
job_config.autodetect = True
# For a file in Google Cloud Storage
# uri = "gs://your-bucket/your-file.json"
# job = client.load_table_from_uri(
#     uri,
#     f"{dataset_id}.{table_id}",
#     job_config=job_config
# )

file_path = file_path
xpath = './/pdv'
encoding = 'ISO-8859-1'

# Parse the XML file using the custom function
df = parse_xml(file_path)
#df = pd.read_xml(file_path, xpath=xpath, encoding=encoding, dtype=str)

table_ref = f"{dataset_id}.{table_id}"

json_data = df.to_json(orient='records', lines=True)

job = client.load_table_from_file(
    io.StringIO(json_data),
    table_ref,
    job_config=job_config
)

job.result()  # Wait for the job to complete

print(f"Loaded {job.output_rows} rows into {table_ref}")



This small app explores the open data made available by French government on prices of different gas in French stations.
An xml file is updated up to every 10 minutes on this page, also hosting a daily table, cumulative past 30 days, and yearly data.
https://www.prix-carburants.gouv.fr/rubrique/opendata/


It provides a map of available stations, as well as some recommendation of stations according to the selected gas type and current location (entered initially by the user)
Geopy library is used for latitude, longitude and distance calculations from the address provided by user
https://geopy.readthedocs.io/en/stable/

To limit the computing, only stations in selected address and adjacent departments were considered for available gas stations, using this listing https://gist.github.com/sunny/13803#file-liste-de-departements-limitrophes-francais-txt

Calculations of actual (driving) distance are calculated using the openrouteservice API https://openrouteservice.org/

in left column (the sidebar, as per Streamlit verbatim)
- A feedback form has been connected to a CRM for ticket creation using Hubspot free account and a private app developer API key.
https://developers.hubspot.com/docs/api/private-apps

- Connections are logged in an sql lite database https://www.sqlite.org/, using anonymised IP addresses * for GDPR compliance. Number of apps usage, as well as usage statistics (to the user ticking the checkbox) are provided.

* Instead of storing the full IP address, we do mask the last octet (for IPv4) or the last 80 bits (for IPv6), similarly to Google Analytics' IP anonymization technique.


A specific script was created to a) parse the xml file into a json file, and b) upload the json file onto BigQuery*.

To achieve this, an Application Default Credentials (ADC) file was created using Google Cloud SDK shell. 
Command 'gcloud auth application-default login' saves an 'application_default_credentials.json', path to which is indicated in environment file.

This small app explores the open data made available by French government on prices of different gas in French stations.
An xml file is updated up to every 10 minutes on this page, also hosting a daily table, cumulative past 30 days, and yearly data.
https://www.prix-carburants.gouv.fr/rubrique/opendata/


It provides a map of available stations, as well as some recommendation of stations according to the selected gas type and current location (entered initially by the user)
Geopy library is used for latitude, longitude and distance calculations from the address provided by user
https://geopy.readthedocs.io/en/stable/

To limit the computing, only stations in selected address and adjacent departments were considered for available gas stations, using this listing https://gist.github.com/sunny/13803#file-liste-de-departements-limitrophes-francais-txt

Calculations of actual (driving) distance are calculated using the openrouteservice API https://openrouteservice.org/

Exploration of the linkage of user feedback to a CRM has been performed using Hubspot free account and a private app developer account https://developers.hubspot.com/docs/api/private-apps 





This small app explores the open data on prices of different gas types in French stations
An xml file is updated every up to every 10 minutes on this page, also hosting a daily table, cumulative past 30 days, and yearly data.
https://www.prix-carburants.gouv.fr/rubrique/opendata/


It provides a map of available stations, as well as some recommendation of stations according to the selected gas type and current location (entered initially by the user)
Geopy library is used for latitude, longitude and distance calculations
https://geopy.readthedocs.io/en/stable/

To limit the calculations, only stations in current and adjacent departments were considered for available gas stations, using this listing https://gist.github.com/sunny/13803#file-liste-de-departements-limitrophes-francais-txt



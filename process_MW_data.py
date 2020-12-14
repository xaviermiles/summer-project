
import csv
import pandas as pd
from shapely.geometry import shape
import shapely.wkt
from sentinelhub import read_data
from image_utils import *

pd.set_option("display.max_rows", None, "display.max_columns", None)  # print entire dataframe

with open('./MW landcover data/lcdb-v50-land-cover-database-version-50-mainland-new-zealand.csv') as csvfile:
    raw_data = pd.read_csv(csvfile)
    raw_data.rename(columns={raw_data.columns[0]: 'Location'}, inplace=True)  # the column name is gibberish otherwise

data2018 = raw_data[['Location', 'Name_2018', 'Class_2018', 'Wetland_18', 'Onshore_18', 'LCDB_UID']]
data2018.columns = ['Location', 'Name', 'Class', 'Wetland', 'Onshore', 'ID']

COUNTRIES_FILE = 'countries.json'
countries_geo = read_data(COUNTRIES_FILE)
# nz_shape = shape(countries_geo["features"][172]["geometry"])


print(data2018.head())
print(data2018.dtypes)
print()
print(shapely.wkt.loads(data2018['Location']))
show_area(shapely.wkt.loads(data2018.loc[0, 'Location']))

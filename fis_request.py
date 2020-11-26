"""
The main purpose of FIS service is to enable users to obtain statistics about satellite data without actually having
to download large amounts of raster imagery. This is most effective when you have a sparse collection of bbox/polygons
and would like to know aggregated values for each of them.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import Polygon

from sentinelhub import FisRequest, BBox, Geometry, CRS, WcsRequest, CustomUrlParam, DataCollection, HistogramType
from sentinelhub.time_utils import iso_to_datetime
from sentinelhub import SHConfig

INSTANCE_ID = 'cae04732-5ed8-40c2-a2c6-461e44f55720'
if INSTANCE_ID:
    config = SHConfig()
    config.instance_id = INSTANCE_ID
else:
    config = None


"""    
Exploring basic statistics
Example is concerned with Sahara deser during a sandstorm in March 2018
"""
sahara_bbox = BBox((1.0, 26.0, 1.3, 25.7), CRS.WGS84)
time_interval = ('201802-01', '2018-05-01')

fis_request = FisRequest(
    data_collection=DataCollection.SENTINEL2_L1C,
    layer='BANDS-S2-L1C',
    geometry_list=[sahara_bbox],
    time=time_interval,
    resolution='60m',
    data_folder='./data',
    config=config
)

# the call concurrently makes calls to FIS for each geometry and collects the responses
# save to avoid redownloading (~30s)
fis_data = fis_request.get_data(save_data=True)

print(fis_data[0]['C0'][0])

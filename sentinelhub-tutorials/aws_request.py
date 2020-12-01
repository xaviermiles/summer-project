"""
Accessing satellite data from AWS with Python
"""

import numpy as np
import matplotlib.pyplot as plt

from sentinelhub import WebFeatureService, BBox, CRS, DataCollection, SHConfig, get_area_info
from sentinelhub import AwsTile, AwsTileRequest


INSTANCE_ID = '2d26c335-1184-42e3-8190-5de92f6c6aac'
if INSTANCE_ID:
    config = SHConfig()
    config.instance_id = INSTANCE_ID
else:
    config = None

search_bbox = BBox(bbox=[46.16, -16.15, 46.51, -15.58], crs=CRS.WGS84)
search_time_interval = ('2017-12-01T00:00:00', '2017-12-15T23:59:59')

wfs_iterator = WebFeatureService(
    search_bbox,
    search_time_interval,
    data_collection=DataCollection.SENTINEL2_L1C,
    maxcc=1.0,
    config=config
)

for tile_info in wfs_iterator:
    print(tile_info)

# extract info which uniquely defines each tile
print(wfs_iterator.get_tiles())

# automatic search with functions from sentinelhub.opensearch, which does not require authentication
for tile_info in get_area_info(search_bbox, search_time_interval, maxcc=0.5):
    print(tile_info)

"""
DOWNLOAD DATA - once we have found correct tiles or products

The AWS index is the last number in the AWS path
"""
tile_id = 'S2A_OPER_MSI_L1C_TL_MTI__20151219T100121_A002563_T38TML_N02.01'
tile_name, time, aws_index = AwsTile.tile_id_to_tile(tile_id)
print(tile_name, time, aws_index)

bands_to_download = ['B8A', 'B10']
metafiles = ['tileInfo', 'preview', 'qi/MSK_CLOUDS_B00']
data_folder = './AwsData'

request = AwsTileRequest(
    tile=tile_name,
    time=time,
    aws_index=aws_index,
    bands=bands_to_download,
    metafiles=metafiles,
    data_folder=data_folder,
    data_collection=DataCollection.SENTINEL2_L1C
)

request.save_data()  # rerunning code won't redownload data unless redownload=True

data_list = request.get_data() # won't redownload since data is already stored on disk

b8a, b10, tile_info, preview, cloud_mask = data_list

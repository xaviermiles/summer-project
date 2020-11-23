import os
import datetime
import numpy as np
import matplotlib.plot as plt

from sentinelhub import SHConfig
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, \
    SentinelHubDownloadClient, DataCollection, bbox_to_dimensions, DownloadRequest
from utils import plot_image

# Setup Sentinelhub API interaction
CLIENT_ID = 'db3d37b7-38e9-44e3-a4e5-00c0e49bdede'
CLIENT_SECRET = '6PI6CClueGo~++WI:32:I@bpz%ZdcQ4i:05c_zF1'

config = SHConfig()

if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

if config.sh_client_id == '' or config.sh_client_secret == '':
    print("LOL Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")

betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]
resolution = 60
betsiboka_bbox = BBox(bbox=betsiboka_bbox, crs=CRS.WGS84)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

print(f'Image shape at {resolution} m resolution: {betsiboka_size} pixels')

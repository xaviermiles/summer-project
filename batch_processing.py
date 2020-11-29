"""
'large_area_utilities' showed how to split a large area into smaller bounding box which can then be obtained using the
requests demonstrated in 'processing_api_request'

Sentinel Hub Batch Processing shows another way to do that:
- takes the geometry of a large area and divides it according to a specified tile grid
- executes processing requests for each tile in the grid and stores results in a given location at AWS S3 storage
Because of the optimised performance, this is significantly faster than running the same process locally

1. Define and create a batch request
2. Analyse a batch request before it is executed
3. Run a batch requests job and check the outcome

CANNOT DO WITHOUT AN ENTERPRISE ACCOUNT
"""

import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import cv2

from sentinelhub import SentinelHubBatch, SentinelHubRequest, Geometry, CRS, DataCollection, MimeType, SHConfig, \
    bbox_to_dimensions


def change_brightness(img, alpha, beta=0):
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)


# 1. Create a batch request
# 1.1 Define a Processing API request
CLIENT_ID = 'ac45629f-3b23-4a8d-a579-0a30bbfbaa0e'
CLIENT_SECRET = 'd,w~E#IpJJ6eTfEUd$A*q<bU5hNivI!jANtt<7WP'
config = SHConfig()
if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

SHAPE_PATH = os.path.join('.', 'data', 'california_crop_fields.geojson')
area_gdf = gpd.read_file(SHAPE_PATH)

# Geometry of entire area
full_geometry = Geometry(area_gdf.geometry.values[0], crs=CRS.WGS84)
# BBox of a test sub-area
test_bbox = Geometry(area_gdf.geometry.values[1], crs=CRS.WGS84).bbox

area_gdf.plot(column='name')

# Check a true-colour satellite image of the entire area
evalscript_true_colour = """
    //VERSION=3
    function setup() {
        return {
            input: [{ 
                bands: ["B02", "B03", "B04"] 
            }],
            output: { bands: 3 }
        }
    }
    
    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request = SentinelHubRequest(
    evalscript=evalscript_true_colour,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    geometry=full_geometry,
    size=(512, 512),
    config=config
)

image = request.get_data()[0]
plt.figure()
plt.imshow(change_brightness(image, 3.5))

# Define an evalscript and time range. This evalscript will return a temporally-inpterpolated stack NDVI values.
# To change the time interval, it must be changed both in the cell and in the evalscript code.
EVALSCRIPT_PATH = os.path.join('.', 'data', 'interpolation_evalscript.js')

with open(EVALSCRIPT_PATH, 'r') as fp:
    evalscript = fp.read()

time_interval = dt.date(year=2020, month=7, day=1), dt.date(year=2020, month=7, day=30)

sentinelhub_test_request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=time_interval
        )
    ],
    responses=[
        SentinelHubRequest.output_response('NDVI', MimeType.TIFF),
        SentinelHubRequest.output_response('data_mask', MimeType.TIFF)
    ],
    bbox=test_bbox,
    size=bbox_to_dimensions(test_bbox, 10),
    config=config
)

results = sentinelhub_test_request.get_data()[0]

print(f'Output data: {list(results)}')

plt.imshow(results['NDVI.tif'][..., 2])

# 1.2 Select a tiling grid
list(SentinelHubBatch.iter_tiling_grids(config=config))

GRID_ID = 1
SentinelHubBatch.get_tiling_grid(GRID_ID, config=config)

# 1.3 Set up an S3 bucket
BUCKET_NAME = 'sail-summer'

# 1.4 Join batch request definition - this won't trigger the actual processing
sentinelhub_request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=time_interval
        )
    ],
    responses=[
        SentinelHubRequest.output_response('NDVI', MimeType.TIFF),
        SentinelHubRequest.output_response("data_mask", MimeType.TIFF)
    ],
    geometry=full_geometry,
    # do not specify size parameter this time
    config=config
)

batch_request = SentinelHubBatch.create(
    sentinelhub_request,
    tiling_grid=SentinelHubBatch.tiling_grid(
        grid_id=GRID_ID,
        resolution=10,
        buffer=(50, 50)
    ),
    bucket_name=BUCKET_NAME,
    # Check documentation for more about output configuration options:
    # output=SentinelHubBatch.output(...)
    description='sentinelhub-py tutorial batch job',
    config=config
)

print(batch_request.__doc__)
print()
print(batch_request.info)


plt.show()

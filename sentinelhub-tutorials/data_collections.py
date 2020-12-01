"""
Tutorial to show how to obtain data from a desired data collection using sentinelhub.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import DataCollection
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, MimeType, bbox_to_dimensions


def change_brightness(img, alpha=1.0, beta=0.0):
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)


for collection in DataCollection.get_available_collections():
    print(collection)

CLIENT_ID = '9d542985-ed95-4357-b740-37596b1d5f70'
CLIENT_SECRET = ')(/q5|#Q[[C:Oy3Is46,kpQYR28ovX9y2@,6UP@D'
config = SHConfig()
if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

# Columbia Glacier, Alaska
glacier_bbox = BBox([-147.8, 60.96, -146.5, 61.38], crs=CRS.WGS84)
glacier_size = (700, 466)
time_interval = '2020-07-15', '2020-07-16'

evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=glacier_bbox,
    size=glacier_size,
    config=config
)

image = request.get_data()[0]

plt.figure()
plt.imshow(change_brightness(image, alpha=3.5))

"""
Switch data collection to Sentinel-1 (SAR), take only data with IW polarisation, and limit to ascending orbits.
"""
print(DataCollection.SENTINEL1_IW_ASC.__doc__)

# use simplified evalscript
evalscript = """
    //VERSION=3

    return [VV, 2 * VH, VV / VH / 100.0, dataMask]
"""

time_interval = '2020-07-06', '2020-07-07'

request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL1_IW_ASC,
            time_interval=time_interval,
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=glacier_bbox,
    size=glacier_size,
    config=config
)

image = request.get_data()[0]

plt.figure()
plt.imshow(change_brightness(image, alpha=3.5))

"""
Sentinel-3 OLCI data collection
"""
evalscript = """
    //VERSION=3

    return [B08, B06, B04]
"""

time_interval = '2020-07-06', '2020-07-07'

request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL3_OLCI,
            time_interval=time_interval,
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=glacier_bbox,
    size=glacier_size,
    config=config
)

image = request.get_data()[0]

plt.figure()
plt.imshow(change_brightness(image, alpha=3.5))

"""
It is possible to define a new data collection.

The most common examples of this are Bring Your Own Data (BYOC) data collections, where uses can bring their own data
and access it with Sentinel Hub service.
- Convert the data to Cloud Optimized GeoTiff (COG) format. Store it in AWS S3 bucket and allow SH to acces it.
- Create a collection in SH which points to the S3 bucket. Within the collection, ingest the tiles from the bucket.

Alternatively, you can take an existing data colelction and create a new data collection from it.
"""
DataCollection.define(
    name='CUSTOM_SENTINEL1',
    api_id='S1GRD',
    wfs_id='DSS3',
    service_url='https://services.sentinel-hub.com',
    collection_type='Sentinel-1',
    sensor_type='C-SAR',
    processing_level='GRD',
    swath_mode='IW',
    polarization='SV',
    resolution='HIGH',
    orbit_direction='ASCENDING',
    timeliness='NRT10m',
    bands=('W',),
    is_timeless=False
)
print(DataCollection.CUSTOM_SENTINEL1)

# define a new BYOC data collection
collection_id = '7453e962-0ee5-4f74-8227-89759fbe9ba9'

byoc = DataCollection.define_byoc(
    collection_id,
    name='SLOVENIA_LAND_COVER',
    is_timeless=True
)
print(byoc.__dir__)

# load data for the defined BYOC data collection
slovenia_bbox = BBox([13.35882, 45.402307, 16.644287, 46.908998], crs=CRS.WGS84)
slovenia_size = bbox_to_dimensions(slovenia_bbox, resolution=240)

evalscript_byoc = """
//VERSION=3
function setup() {
  return {
    input: ["lulc_reference"],
    output: { bands: 3 }
  };
}

var colorDict = {
  0: [255/255, 255/255, 255/255],
  1: [255/255, 255/255, 0/255],
  2: [5/255, 73/255, 7/255],
  3: [255/255, 165/255, 0/255],
  4: [128/255, 96/255, 0/255],
  5: [6/255, 154/255, 243/255],
  6: [149/255, 208/255, 252/255],
  7: [150/255, 123/255, 182/255],
  8: [220/255, 20/255, 60/255],
  9: [166/255, 166/255, 166/255],
  10: [0/255, 0/255, 0/255]
}

function evaluatePixel(sample) {
  return colorDict[sample.lulc_reference];
}
"""

byoc_request = SentinelHubRequest(
    evalscript=evalscript_byoc,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=byoc
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=slovenia_bbox,
    size=slovenia_size,
    config=config
)

byoc_data = byoc_request.get_data()
plt.figure()
plt.imshow(byoc_data[0])

# create a data collection with a different service_url argument. this will collect data from MUNDI deployment (rather
# than SH deployment, which is the default)
s2_12a_mundi = DataCollection.define_from(
    DataCollection.SENTINEL2_L2A,
    'SENTINEL2_L2A_MUNDI',
    service_url='https://shservices.mundiwebservices.com'
)
print(s2_12a_mundi.__dir__)

time_interval = '2020-06-01', '2020-07-01'

evalscript = """
    //VERSION=3
    
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands : 3
            }
        };
    }
    
    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=s2_12a_mundi,
            time_interval=time_interval,
            mosaicking_order='leastCC'
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=slovenia_bbox,
    size=slovenia_size,
    config=config
)

image = request.get_data()[0]

plt.imshow(change_brightness(image, alpha=3.5))

# show all plots
plt.show()

from sentinelhub import SHConfig

# Setup Sentinelhub API interaction
CLIENT_ID = '9d542985-ed95-4357-b740-37596b1d5f70'
CLIENT_SECRET = ')(/q5|#Q[[C:Oy3Is46,kpQYR28ovX9y2@,6UP@D'

config = SHConfig()

if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

if config.sh_client_id == '' or config.sh_client_secret == '':
    print("LOL Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")


import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, \
    SentinelHubDownloadClient, DataCollection, bbox_to_dimensions, DownloadRequest

betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]
resolution = 60
betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

print(f'Image shape at {resolution} m resolution: {betsiboka_size} pixels')


def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.add(hsv[:, :, 2], value)
    out_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out_img


# Example 1: True colour (PNG) on a specific date
evalscript_true_colour = """
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

request_true_colour = SentinelHubRequest(
    evalscript=evalscript_true_colour,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=('2020-06-12', '2020-06-13')
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)

true_colour_imgs = request_true_colour.get_data()
print(f'Returned data is of type = {type(true_colour_imgs)} and length {len(true_colour_imgs)}.')
print(f'Single element in the list is of type {type(true_colour_imgs[-1])} and has shape {true_colour_imgs[-1].shape}.')

image = true_colour_imgs[0]
print(f'Image type: {image.dtype}')

# plot function
# factor 3.5 to increase brightness
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cv2.convertScaleAbs(image, alpha=3.5), interpolation='none')
ax.set_title('Brightened image')
#plt.show()

histogram, bins_edges = np.histogram(np.ndarray.flatten(image), bins=256)
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(image.flatten(), bins=256)
#plt.show()


# Example 1.1 Adding cloud mask data_collection
"""
It is possilbe to obtain cloud masks when requesting Sentinel-2 data by using
the cloud mask band (CLM) or cloud probabilities band (CLP).
Also, the factor for increasing the image brightness can be included in the
evalscript.
"""

evalscript_clm = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "CLM"],
    output: { bands: 3 }
  }
}

function evaluatePixel(sample) {
  if (sample.CLM == 1) {
    return [0.75 + sample.B04, sample.B03, sample.B02]
  }
  return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02];
}
"""

request_true_colour = SentinelHubRequest(
    evalscript=evalscript_clm,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=('2020-06-12', '2020-06-13')
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)

data_with_cloud_mask = request_true_colour.get_data()
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(data_with_cloud_mask[0])


# Example 2: True colour mosaic of least cloudy acquisitions
"""
We will provide a month-long interval, order the image w.r.t. the cloud
coverage on the tile level (leastCC parameter), and mosaic them in the specified
order.
"""

request_true_colour = SentinelHubRequest(
    evalscript=evalscript_clm,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=('2020-06-01', '2020-06-30'),  # month-long interval
            mosaicking_order='leastCC'
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(request_true_colour.get_data()[0])


# Example 3: All Sentinel-2's raw band values
"""
This evalscript will return all Sentinel-2 spectral bands with raw values.

Downloading raw digital numbers in INT16 format rather than reflectances in
FLOAT32 format means that far less data is downloaded. The digital numbers are
in [0, 10000] so scaling is necessary.
"""

evalscript_all_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
                units: "DN"
            }],
            output: {
                bands: 13,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B01,
                sample.B02,
                sample.B03,
                sample.B04,
                sample.B05,
                sample.B06,
                sample.B07,
                sample.B08,
                sample.B09,
                sample.B10,
                sample.B11,
                sample.B12];
    }
"""

request_all_bands = SentinelHubRequest(
    evalscript=evalscript_all_bands,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=('2020-06-01', '2020-06-30'),
            mosaicking_order='leastCC'
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)

all_bands_response = request_all_bands.get_data()

# Image showing the SWIR band B12
# Factor 1/1e4 due to the DN band values in the range [0, 10000]
# Factor 3.5 to increase brightness
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cv2.normalize(all_bands_response[0][:, :, 12],
                        None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_32F))
ax.set_title("SWIR band B12")

# From raw bands we can also construct a False-Colour Image
# False colour image is (B03, B04, B08)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cv2.normalize(all_bands_response[0][:, :, [2,3,7]],
                        None, alpha=0, beta=3.5, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_32F))
ax.set_title("False colour image")


# Example 4: Sve downloaded data to disk and read it from disk
"""
All downloaded dat can be saved to disk and later read from it. This can be done
when requesting the image by specifying the file path to the 'data_folder'
argument in the request's constructor. When executing get_data(...), set
the 'save_data' argument to True.

For all future requests for data, the request will first check the provided
location to see if the data is already there, unless you explicitly demand to
redownload the data.
"""

request_all_bands = SentinelHubRequest(
    data_folder='test_dir',
    evalscript=evalscript_all_bands,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=('2020-06-01', '2020-06-30'),
            mosaicking_order='leastCC'
    )],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)
all_bands_img = request_all_bands.get_data(save_data=True)

print(f'The output directory has been created and a TIFF file with all 13 bands was saved into ' \
      'the following structure:\n')
for folder, _, filenames in os.walk(request_all_bands.data_folder):
    for filename in filenames:
        print(os.path.join(folder, filename))

# try to redownload the data_folder
all_bands_img_from_disk = request_all_bands.get_data()
# force the redownload
all_bands_img_redownload = request_all_bands.get_data(redownload=True)

# Example 4.1: Save downloaded data directly to disk
"""
get_data(...) returns a list of NumPy arrays and can save to disk, whereas
save_data(...) just saves the data directly to the disk.
"""
request_all_bands.save_data()

print(f'The output directory has been created and a tiff file with all 13 bands was saved into ' \
      'the following structure:\n')
for folder, _, filenames in os.walk(request_all_bands.data_folder):
    for filename in filenames:
        print(os.path.join(folder, filename))


# Example 5: Other Data Collections
"""
The sentinelhub package supports various data collections. The previous examples
have used 'DataCollection.SENTINEL2_L1C'.
"""
print('Supported DataCollections:\n')
for collection in DataCollection.get_available_collections():
    print(collection)

# try Digital Elevation Model (DEM) type:
# use FLOAT32 since the values can be negative (below sea level)
evalscript_dem = '''
//VERSION=3
function setup() {
  return {
    input: ["DEM"],
    output: {
      id: "default",
      bands: 1,
      sampleType: SampleType.FLOAT32
    }
  }
}

function evaluatePixel(sample) {
  return [sample.DEM]
}
'''

dem_request = SentinelHubRequest(
    evalscript=evalscript_dem,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.DEM,
            time_interval=('2020-06-12', '2020-06-13')
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)
dem_data = dem_request.get_data()

# plot DEM map
# vmin = 0 (cutoff at sea level - 0m)
# vmax = 120 (cutoff all values higher than 120m)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(dem_data[0], cmap=plt.cm.Greys_r, vmin=0, vmax=120)


# Example 6: Multi-response request type
"""
The API enables downloading multiple files in one response, packaged together in
a TAR archive.

We will download the image in the form of digital numbers (DN) as a UINT16 TIFF
file and download the inputMetadata which contains a normalisation factor value
in JSON format.
Then we will convert the INT16 DNs to get the FLOAT32 reflectances.
"""

evalscript = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"],
                units: "DN"
            }],
            output: {
                bands: 3,
                sampleType: "INT16"
            }
        };
    }

    function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
        outputMetadata.userData = { "norm_factor":  inputMetadata.normalizationFactor }
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request_multitype = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=('2020-06-01', '2020-06-30'),
            mosaicking_order='leastCC'
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF),
        SentinelHubRequest.output_response('userdata', MimeType.JSON)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)
multi_data = request_multitype.get_data()[0]
print("multi_data keys:", multi_data.keys())

# normalize image
img = multi_data['default.tif']
norm_factor = multi_data['userdata.json']['norm_factor']

img_float32 = img * norm_factor
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_float32)  # not brightened

plt.show()

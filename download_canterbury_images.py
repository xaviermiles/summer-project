"""
Aim is to download some images from around Christchurch and see resolution etc.
"""

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from shapely.geometry import shape

from sentinelhub import SHConfig, MimeType, CRS, BBox, read_data, SentinelHubRequest, \
    SentinelHubDownloadClient, DataCollection, bbox_to_dimensions, DownloadRequest
from sentinelhub import BBoxSplitter, OsmSplitter, TileSplitter, CustomGridSplitter, UtmZoneSplitter, UtmGridSplitter

from image_utils import *  # misc custom image functions

CLIENT_ID = '1089387f-e062-426a-a7ec-9c44d7f7a3c0'
CLIENT_SECRET = 'ADB|+iYvOF23Uz5lQ<rhz3Of+NO)TZ{]b*Fe)#D.'
INSTANCE_ID = 'cae04732-5ed8-40c2-a2c6-461e44f55720'

config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET
config.instance_id = INSTANCE_ID


# Get the Canterbury Polygon from the JSON file
POLYGON_INPUT_FILE = './canterbury_images/canterbury_shape.json'

geo_json = read_data(POLYGON_INPUT_FILE)
canterbury_shape = shape(geo_json["features"][0]["geometry"])

# Split area into smaller BBoxes using different splitters
# not BBoxSplitter, picked OsmSplitter as it is relatively eady-to-use

SAVE_IMAGES = True
OSM_ZOOM_LEVEL = 8

osm_splitter = OsmSplitter([canterbury_shape], CRS.WGS84, zoom_level=OSM_ZOOM_LEVEL)
show_splitter(osm_splitter, title=f"OsmSplitter (zoom={osm_splitter.zoom_level})")

tile_splitter = TileSplitter(
    [canterbury_shape],
    CRS.WGS84,
    time_interval=('2020-06-01', '2020-06-30'),
    tile_split_shape=3,
    data_collection=DataCollection.SENTINEL2_L1C,
    config=config
)
show_splitter(tile_splitter, title='TileSplitter')


osm_bbox_list = osm_splitter.get_bbox_list()
osm_info_list = osm_splitter.get_info_list()
print(osm_bbox_list)
print(osm_bbox_list[0])
print(osm_info_list)


evalscript_true_colour = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: { bands: 3 }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""


def get_true_colour_request(bbox, resolution):
    size = bbox_to_dimensions(bbox, resolution)

    return SentinelHubRequest(
        evalscript=evalscript_true_colour,
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
        bbox=bbox,
        size=size,
        config=config
    )


# Configure group of requests into list
list_of_requests = [get_true_colour_request(bbox, resolution=120) for bbox in osm_bbox_list]
list_of_requests = [request.download_list[0] for request in list_of_requests]

# download data with multiple threads
canterbury_images = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5)

if SAVE_IMAGES:
    for idx, image in enumerate(canterbury_images):
        # Get BBox information
        minx, miny, maxx, maxy = osm_bbox_list[idx]
        bbox_string = '{:.2f}_{:.2f}_{:.2f}_{:.2f}'.format(minx, miny, maxx, maxy)

        # Increase brightness
        brightened_image = change_brightness(image, 3.5)

        # Save image to disk
        img_to_save = Image.fromarray(brightened_image)
        img_to_save.save(f'canterbury_images/osm_zoom{osm_splitter.zoom_level}_{bbox_string}.jpeg')


# PLOTTING
ncols = 4; nrows = 3
bbox_size = bbox_to_dimensions(osm_bbox_list[0], resolution=40)
aspect_ratio = bbox_size[0] / bbox_size[1]
subplot_kw = {'xticks': [], 'yticks': [], 'frame_on': True}

# fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
#                         figsize=(5 * ncols * aspect_ratio, 5 * nrows),
#                         subplot_kw=subplot_kw)
# for idx, image in enumerate(canterbury_images):
#     ax = axs[idx // ncols][idx % ncols]
#     ax.imshow(change_brightness(image, 2, 0))
#     ax.set_title(f'{slots[idx][0]}  -  {c[idx][1]}', fontsize=10)
# plt.tight_layout()
plt.show()

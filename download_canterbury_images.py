"""
Aim is to download some images from around Canterbury to adequate resolution
"""

import os
from timeit import default_timer as timer
from PIL import Image
from shapely.geometry import shape
from sentinelhub import SHConfig, MimeType, read_data, SentinelHubRequest, SentinelHubDownloadClient, DataCollection, \
    bbox_to_dimensions, OsmSplitter, TileSplitter

from image_utils import *  # misc custom image functions

# Configure connection to sentinelhub website
CLIENT_ID = '1089387f-e062-426a-a7ec-9c44d7f7a3c0'
CLIENT_SECRET = 'ADB|+iYvOF23Uz5lQ<rhz3Of+NO)TZ{]b*Fe)#D.'
INSTANCE_ID = 'cae04732-5ed8-40c2-a2c6-461e44f55720'

config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET
config.instance_id = INSTANCE_ID


SAVE_IMAGES = True
OSM_ZOOM_LEVEL = 11  # refer to https://wiki.openstreetmap.org/wiki/Zoom_levels
RESOLUTION = 10  # resolution of downloaded images in metres


# Get the Canterbury Polygon from a JSON file
POLYGON_INPUT_FILE = './canterbury_images/canterbury_shape.json'

geo_json = read_data(POLYGON_INPUT_FILE)
canterbury_shape = shape(geo_json["features"][0]["geometry"])

# Split area into smaller BBoxes using splitters
# Using OsmSplitter as it is relatively easy-to-use and understandable
osm_splitter = OsmSplitter([canterbury_shape], CRS.WGS84, zoom_level=OSM_ZOOM_LEVEL)
show_splitter(osm_splitter, title=f"OsmSplitter (zoom={OSM_ZOOM_LEVEL})")

osm_bbox_list = osm_splitter.get_bbox_list()
osm_info_list = osm_splitter.get_info_list()
# print('Some examples of the smaller bboxes:')
# for bbox, info in zip(osm_bbox_list[:5], osm_info_list[:5]):
#     print(f'bbox: {bbox}\ninfo: {info}')

# tile_splitter = TileSplitter(
#     [canterbury_shape],
#     CRS.WGS84,
#     time_interval=('2020-06-01', '2020-06-30'),
#     tile_split_shape=3,
#     data_collection=DataCollection.SENTINEL2_L1C,
#     config=config
# )
# show_splitter(tile_splitter, title='TileSplitter')


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
        return [2 * sample.B04, 2 * sample.B03, 2 * sample.B02];
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
                mosaicking_order='leastCC'  # least cloud coverage
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
list_of_requests = [get_true_colour_request(bbox, resolution=RESOLUTION)
                    for bbox in osm_bbox_list]
list_of_requests = [request.download_list[0] for request in list_of_requests]

# download data (using multiple threads)
print(f'downloading images now...\n...{len(list_of_requests)} requests...')
start = timer()
canterbury_images = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5)
end = timer()
time_elapsed = end - start
print('...finished ({:.2f} sec)'.format(time_elapsed))

if SAVE_IMAGES:
    for idx, image in enumerate(canterbury_images):
        # Get BBox information
        minx, miny, maxx, maxy = osm_bbox_list[idx]
        bbox_string = '{:05.2f}_{:05.2f}__{:05.2f}_{:05.2f}'.format(minx, miny, maxx, maxy)

        # Make appropriate folder to save to (if not already present)
        if not os.path.exists(f'canterbury_images/osm{OSM_ZOOM_LEVEL}'):
            os.mkdir(f'canterbury_images/osm{OSM_ZOOM_LEVEL}')

        # Save image to disk
        img_to_save = Image.fromarray(image)
        img_to_save.save(f"canterbury_images/osm{OSM_ZOOM_LEVEL}/osm{OSM_ZOOM_LEVEL}_res{RESOLUTION}m"
                         f"__{bbox_string}.jpeg")


# PLOTTING
# ncols = 4; nrows = 3
# bbox_size = bbox_to_dimensions(osm_bbox_list[0], resolution=40)
# aspect_ratio = bbox_size[0] / bbox_size[1]
# subplot_kw = {'xticks': [], 'yticks': [], 'frame_on': True}

# fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
#                         figsize=(5 * ncols * aspect_ratio, 5 * nrows),
#                         subplot_kw=subplot_kw)
# for idx, image in enumerate(canterbury_images):
#     ax = axs[idx // ncols][idx % ncols]
#     ax.imshow(change_brightness(image, 2, 0))
#     ax.set_title(f'{slots[idx][0]}  -  {c[idx][1]}', fontsize=10)
# plt.tight_layout()
plt.show()

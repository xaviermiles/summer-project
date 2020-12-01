"""
Aim is to download some images from around Christchurch and see resolution etc.
"""

import itertools
import cv2
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon
from mpl_toolkits.basemap import Basemap
from shapely.geometry import shape, Polygon, MultiLineString

from sentinelhub import SHConfig, MimeType, CRS, BBox, read_data, SentinelHubRequest, \
    SentinelHubDownloadClient, DataCollection, bbox_to_dimensions, DownloadRequest
from sentinelhub import BBoxSplitter, OsmSplitter, TileSplitter, CustomGridSplitter, UtmZoneSplitter, UtmGridSplitter

CLIENT_ID = '1089387f-e062-426a-a7ec-9c44d7f7a3c0'
CLIENT_SECRET = 'ADB|+iYvOF23Uz5lQ<rhz3Of+NO)TZ{]b*Fe)#D.'

config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET
if config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")


def change_brightness(img, alpha, beta=0):
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)


def show_splitter(splitter, alpha=0.2, area_buffer=0.2, show_legend=False, title=""):
    area_bbox = splitter.get_area_bbox()
    minx, miny, maxx, maxy = area_bbox
    lng, lat = area_bbox.middle
    w, h = maxx - minx, maxy - miny
    minx = minx - area_buffer * w
    miny = miny - area_buffer * h
    maxx = maxx + area_buffer * w
    maxy = maxy + area_buffer * h

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    base_map = Basemap(projection='mill', lat_0=lat, lon_0=lng, llcrnrlon=minx, llcrnrlat=miny,
                       urcrnrlon=maxx, urcrnrlat=maxy, resolution='l', epsg=4326)
    base_map.drawcoastlines(color=(0, 0, 0, 0))

    area_shape = splitter.get_area_shape()
    if isinstance(area_shape, Polygon):
        area_shape = [area_shape]  # necessary if there is only one Polygon
    for polygon in area_shape:
        if isinstance(polygon.boundary, MultiLineString):
            for linestring in polygon.boundary:
                ax.add_patch(plt_polygon(np.array(linestring), closed=True, facecolor=(0, 0, 0, 0), edgecolor='red'))
        else:
            ax.add_patch(plt_polygon(np.array(polygon.boundary), closed=True, facecolor=(0, 0, 0, 0), edgecolor='red'))

    bbox_list = splitter.get_bbox_list()
    info_list = splitter.get_info_list()

    cm = plt.get_cmap('jet', len(bbox_list))
    legend_shapes = []
    for i, (bbox, info) in enumerate(zip(bbox_list, info_list)):
        wgs84_bbox = bbox.transform(CRS.WGS84).get_polygon()

        tile_colour = tuple(list(cm(i))[:3] + [alpha])
        ax.add_patch(plt_polygon(np.array(wgs84_bbox), closed=True, facecolor=tile_colour, edgecolor='green'))

        if show_legend:
            legend_shapes.append(plt.Rectangle((0, 0), 1, 1, fc=cm(i)))

    if show_legend:
        legend_names = []
        for info in info_list:
            legend_name = '{},{}'.format(info['index_x'], info['index_y'])

            for prop in ['grid_index', 'tile']:
                if prop in info:
                    legend_name = '{},{}'.format(info[prop], legend_name)

            legend_names.append(legend_name)

        plt.legend(legend_shapes, legend_names)

    plt.title(title)
    plt.tight_layout()


# Get the Canterbury Polygon from the JSON file
POLYGON_INPUT_FILE = './canterbury_images/canterbury_shape.json'

geo_json = read_data(POLYGON_INPUT_FILE)
canterbury_shape = shape(geo_json["features"][0]["geometry"])

# Split area into smaller BBoxes using different splitters
# not BBoxSplitter, picked OsmSplitter as it is relatively eady-to-use

osm_splitter = OsmSplitter([canterbury_shape], CRS.WGS84, zoom_level=8)
show_splitter(osm_splitter, title=f"OsmSplitter (zoom={osm_splitter.zoom_level})")


osm_bbox_list = osm_splitter.get_bbox_list()
osm_info_list = osm_splitter.get_info_list()
print(osm_bbox_list)
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
        data_folder='canterbury_images',
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

# download data with multiple threads, then save to disk (if desired/SAVE_IMAGES==True)
canterbury_images = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5)
SAVE_IMAGES = True
if SAVE_IMAGES:
    for idx, image in enumerate(canterbury_images):
        brightened_image = change_brightness(image, 3.5)
        img_to_save = Image.fromarray(brightened_image)
        img_to_save.save(f'canterbury_images/osm_zoom{osm_splitter.zoom_level}_{idx}.jpeg')


# PLOTTING
ncols = 4; nrows = 3
size = bbox_to_dimensions(osm_bbox_list[0], resolution=40)
aspect_ratio = size[0] / size[1]
subplot_kw = {'xticks': [], 'yticks': [], 'frame_on': True}

fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                        figsize=(5 * ncols * aspect_ratio, 5 * nrows),
                        subplot_kw=subplot_kw)
# for idx, image in enumerate(canterbury_images):
#     ax = axs[idx // ncols][idx % ncols]
#     ax.imshow(change_brightness(image, 2, 0))
#     ax.set_title(f'{slots[idx][0]}  -  {c[idx][1]}', fontsize=10)
# plt.tight_layout()
plt.show()

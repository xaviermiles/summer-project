"""
Imports New Zealand shape from JSON file, splits this into manageable bounding boxes, and sends a request to SentinelHub
API to get set of satellite images.
"""

import os
from timeit import default_timer as timer

from PIL import Image
from sentinelhub import SHConfig, MimeType, read_data, SentinelHubRequest, SentinelHubDownloadClient, DataCollection, \
    bbox_to_dimensions, OsmSplitter
from shapely.geometry import shape

import image_utils # misc custom image functions
import api_config  # includes API keys for sentinelhub

# Configure connection to sentinelhub website
config = SHConfig()
config.sh_client_id = api_config.client_id
config.sh_client_secret = api_config.client_secret
config.instance_id = api_config.instance_id


def get_true_colour_request(bbox, resolution, evalscript):
    """

    Args:
        bbox:
        resolution:
        evalscript:

    Returns: SentinelHubRequest configured according to provided arguments. Currently gets images from Mar-Apr 2020 and
        pieces together the images with the least cloud coverage.

    """
    size = bbox_to_dimensions(bbox, resolution)

    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,  # which satellite?
                time_interval=('2020-03-01', '2020-04-30'),  # two month long interval
                mosaicking_order='leastCC'  # least cloud-coverage
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.PNG)
        ],
        bbox=bbox,
        size=size,
        config=config
    )


def download_images(geo_shape, osm_zoom_level, resolution, out_folder='', plot_splitting=False):
    """

    Args:
        geo_shape: geometry (eg Polygon, MultiPolygon) generated using shapely.geometry
        osm_zoom_level: int, refer to https://wiki.openstreetmap.org/wiki/Zoom_levels
        resolution: float, resolution of desired images in metres. To see valid values, refer to
            https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/
        out_folder: string, folder to write retrieved images into. If empty, images are not saved.
        plot_splitting: boolean

    Returns: Null. Writes images to file and maybe produces plots (depending on arguments).

    """
    osm_splitter = OsmSplitter([geo_shape], crs=CRS.WGS84, zoom_level=osm_zoom_level)
    osm_bbox_list = osm_splitter.get_bbox_list()

    if plot_splitting:
        image_utils.show_splitter(osm_splitter, title=f"OsmSplitter (zoom={osm_zoom_level})")

    # This evalscript tells sentinelhub to fetch the true colour images
    evalscript_true_colour = """"
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
            return [3 * sample.B04, 3 * sample.B03, 3 * sample.B02];
        }
    """

    # Configure group of requests into list
    list_of_requests = [get_true_colour_request(bbox, resolution=resolution, evalscript=evalscript_true_colour)
                        for bbox in osm_bbox_list]
    list_of_requests = [request.download_list[0] for request in list_of_requests]

    # Download data (using multiple threads)
    example_dimensions = bbox_to_dimensions(osm_bbox_list[0], resolution)
    print(f'downloading images now...\n'
          f'...{len(list_of_requests)} requests...\n'
          f'...each of about {example_dimensions} pixel dimensions...')
    start = timer()
    images = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5)
    end = timer()
    time_elapsed = end - start
    print('...finished ({:.2f} sec)'.format(time_elapsed))

    if out_folder:
        for idx, image in enumerate(images):
            # Get BBox information
            minx, miny, maxx, maxy = osm_bbox_list[idx]
            bbox_string = '{:05.2f}_{:05.2f}__{:05.2f}_{:05.2f}'.format(minx, miny, maxx, maxy)

            # Create folder to save to (if it doesn't already exist)
            if not os.path.exists(f'{out_folder}/osm{osm_zoom_level}'):
                os.mkdir(f'{out_folder}/osm{osm_zoom_level}')

            # Save image to folder
            img_to_save = Image.fromarray(image)
            img_to_save.save(f"{out_folder}/osm{osm_zoom_level}/osm{osm_zoom_level}_res{resolution}m"
                             f"__{bbox_string}.jpeg")


# Get NZ shape
COUNTRIES_FILE = 'countries.json'
countries_geo = read_data(COUNTRIES_FILE)
nz_shape = shape(countries_geo["features"][172]["geometry"])

# Download and save NZ satellite images
download_images(nz_shape, osm_zoom_level=11, resolution=10, out_folder='nz_images', plot_splitting=False)

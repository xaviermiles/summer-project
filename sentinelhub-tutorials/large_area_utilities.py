"""
Focuses on importing large areas, when there is not enough working memory to
do the entire operation at once. The sentinelhub package enables easily
splitting the area into smaller bounding boxes.
"""

import itertools

import numpy as np
from shapely.geometry import shape, Polygon, MultiLineString

from sentinelhub import SHConfig
from sentinelhub import BBoxSplitter, OsmSplitter, TileSplitter, CustomGridSplitter, UtmZoneSplitter, UtmGridSplitter
from sentinelhub import BBox, read_data, CRS, DataCollection

# imports for visualisations
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon
from mpl_toolkits.basemap import Basemap

INPUT_FILE = './data/Hawaii.json'

geo_json = read_data(INPUT_FILE)
hawaii_area = shape(geo_json["features"][0]["geometry"])
print(type(hawaii_area))


def show_area(area_shape, area_buffer=0.3):
    """
    Args:
        area_shape:
        area_buffer:

    Returns: None (plotting function)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    minx, miny, maxx, maxy = area_shape.bounds
    lng, lat = (minx + maxx) / 2, (miny + maxy) / 2

    m = Basemap(projection='ortho', lat_0=lat, lon_0=lng, resolution='l')
    m.drawcoastlines()
    m.bluemarble()

    if isinstance(area_shape, Polygon):
        area_shape = [area_shape]
    for polygon in area_shape:
        # x, y = np.array(polygon.boundary)[0]
        m_poly = []
        for x, y in np.array(polygon.boundary):
            m_poly.append(m(x, y))
        ax.add_patch(plt_polygon(np.array(m_poly), closed=True, facecolor='red', edgecolor='red'))

    plt.tight_layout()
    plt.show()


show_area(hawaii_area)

"""
We want to split the area into smaller bounding boxes which can then be used to obtain data with WMS/WCS requests.
There are different ways to split the area implemented in sentinelhub.

The first way is to split the bounding box into smaller parts of equal size. For inputs we must provide: list of 
geometries, their CRS, and an int/tuple specifying how many parts the bounding box will be split into.
"""
bbox_splitter = BBoxSplitter([hawaii_area], CRS.WGS84, (5, 4))  # will produce 5x4 grid of bboxes

print('Area bounding box: {}\n'.format(bbox_splitter.get_area_bbox().__repr__()))

bbox_list = bbox_splitter.get_bbox_list()
info_list = bbox_splitter.get_info_list()

print('Each bounding box also has some info about how it was created.\nExample:\n'
      'bbox: {}\ninfo: {}\n'.format(bbox_list[0].__repr__(), info_list[0]))

# get a list of geometries
geometry_list = bbox_splitter.get_geometry_list()
print(geometry_list[0])


# visualise the splits
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
    plt.show()


show_splitter(bbox_splitter, show_legend=True, title="BBoxSplitter")

# reduce the bboxes so that they do not include area outside the geomtry
bbox_splitter_reduced = BBoxSplitter([hawaii_area], CRS.WGS84, (5, 4), reduce_bbox_sizes=True)

show_splitter(bbox_splitter_reduced, show_legend=True, title="BBoxSplitter (finer)")

"""
The second option is to have a splitting grid idpt from the given geometries. This means that if the geometries change 
slightly, the grid will (likely) stay the same. The splitter below implements Open Street Map's (OSM) grid.

The obtained Sentinel-2 tiles intersect each other, so this splitter is only useful if we are analysing data on level
of the original satellite tiles.
"""
osm_splitter = OsmSplitter([hawaii_area], CRS.WGS84, zoom_level=10)

print(repr(osm_splitter.get_bbox_list()[0]))
print(osm_splitter.get_info_list()[0])

show_splitter(osm_splitter, show_legend=True, title="OsmSplitter")

"""
The third option is to work on a level of satellite tiles and split them using TileSplitter. This requires an instance
ID to utilise Sentinel Hub WFS.
"""

INSTANCE_ID = 'cae04732-5ed8-40c2-a2c6-461e44f55720'
if INSTANCE_ID:
    config = SHConfig()
    config.instance_id = INSTANCE_ID
else:
    config = None

tile_splitter = TileSplitter(
    [hawaii_area],
    CRS.WGS84,
    ('2017-10-01', '2018-03-01'),
    data_collection=DataCollection.SENTINEL2_L1C,
    config=config
)

tile_bbox_list = tile_splitter.get_bbox_list()

print()
print(len(tile_bbox_list))
print(tile_bbox_list[0].__repr__())
print(tile_splitter.get_info_list()[0])
print()
print(tile_splitter.get_bbox_list(crs=CRS.WGS84)[0])

show_splitter(tile_splitter, show_legend=True, title="Satellite tiles")

finer_tile_splitter = TileSplitter(
    [hawaii_area],
    CRS.WGS84,
    ('2017-10-01', '2018-03-01'),
    tile_split_shape=(7, 3),
    data_collection=DataCollection.SENTINEL2_L1C,
    config=config
)

show_splitter(finer_tile_splitter, show_legend=False, title="Satellite tiles (finer)")

"""
The fourth option is to split on a custom grid. The polygons outside of the given collection of bounding boxes will not
affect the tiling.

The example below splits bboxes into integer value latitude and longitude.
"""
bbox_grid = [BBox((x, y, x + 1, y + 1), CRS.WGS84) for x, y in itertools.product(range(-159, -155), range(18, 23))]
print(bbox_grid)

custom_grid_splitter = CustomGridSplitter([hawaii_area], CRS.WGS84, bbox_grid)
show_splitter(custom_grid_splitter, show_legend=True, title="Custom grid")

finer_custom_grid_splitter = CustomGridSplitter([hawaii_area], CRS.WGS84, bbox_grid,
                                                bbox_split_shape=(2, 3), reduce_bbox_sizes=True)
show_splitter(finer_custom_grid_splitter, show_legend=True, title="Custom grid (finer)")

"""
The final option is to split into UTM grid zones or to the UTM Military Grid Reference System.

The 'reduce_bbox_sizes' argument is not available for these splitters as the size of bboxes is specified (in metres).
"""
utm_zone_splitter = UtmZoneSplitter([hawaii_area], CRS.WGS84, (50000, 50000))

show_splitter(utm_zone_splitter, show_legend=True, title="UTM Zones")
print(utm_zone_splitter.get_bbox_list())

utm_grid_splitter = UtmGridSplitter([hawaii_area], CRS.WGS84, (50000, 50000))

show_splitter(utm_grid_splitter, show_legend=True, title="UTM Grids")

"""
Misc functions to be used when collecting images via sentinelhub package
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon, MultiLineString

from sentinelhub import CRS


def change_brightness(img, alpha, beta=0):
    """
    Alter brightness of image, while keeping the values within the valid ranges (eg 0-255).
    Alpha (multiplicative term) must be specified, whereas beta (additive term) is zero by default.
    """
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)


def show_area(area_shape, area_buffer=0.3):
    """
    Plots the area described by a polygon on a globe/world map
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    print(tuple(map(float, area_shape.bounds)))
    print(tuple(map(lambda bound: (1 + area_buffer) * float(bound), area_shape.bounds)))
    minx, miny, maxx, maxy = tuple(map(lambda bound: (1 + area_buffer) * float(bound), area_shape.bounds))
    lng, lat = (minx + maxx) / 2, (miny + maxy) / 2
    print(lng, lat)

    m = Basemap(projection='ortho', lat_0=lat, lon_0=lng, resolution='l')
    m.drawcoastlines()
    m.bluemarble()

    if isinstance(area_shape, Polygon):
        area_shape = [area_shape]
    for polygon in area_shape:
        # print(polygon)
        # x, y = np.array(polygon.boundary)[0]
        m_poly = []
        for x, y in np.array(polygon.boundary):
            m_poly.append(m(x, y))
        ax.add_patch(plt_polygon(np.array(m_poly), closed=True, facecolor='red', edgecolor='red'))

    plt.tight_layout()
    plt.show()


def show_splitter(splitter, alpha=0.2, area_buffer=0.2, show_legend=False, title=""):
    """
    Plots the bboxes which will be constructed from the polygon
    """
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

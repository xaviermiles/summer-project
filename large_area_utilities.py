"""
Focuses on importing large areas, when there is not enough working memory to
do the entire operation at once. The sentinelhub package enables easily
splitting the area into smaller bounding boxes.
"""

import itertools

import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon, MultiLineString

#from sentinelhub import SHConfig#, BBoxSplitter, OsmSplitter, TileSplitter, CustomGridSplitter, UtmZoneSplitter, UtmGridSplitter
#from sentinelhub import BBox, read_data, CRS, DataCollection

# imports for visualisations
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon

#import cartopy

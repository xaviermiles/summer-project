"""
Import landcover data from Manaaki Whenua files and transform polygons into useable annotations for the given set of
images. Download from:
https://lris.scinfo.org.nz/layer/104400-lcdb-v50-land-cover-database-version-50-mainland-new-zealand/

WARNING: requires that the images folder only contains appropriate JPEG files (no other files or folders), and these
should follow a specific naming convention which contains the WGS84 bounding-box coordinates of the images.
"""

import json
import ntpath
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from shapely import wkt
from shapely.geometry import box


def check_for_overlap(bbox, polygon_info):
    """
    Args:
        bbox: shapely.geometry box object, corresponding to the bounding box of a satellite image
        polygon_info: row of data2018

    Returns: Boolean indicating whether any part of the Polygon is contained within the bounding box

    """
    area_in_bbox = bbox.intersection(polygon_info['Location'])

    if not area_in_bbox.is_empty:
        return True
    else:
        return False


def get_overlap(bbox, plygn):
    intersection = bbox.intersection(plygn)

    if intersection.geom_type == 'Polygon':
        return intersection
    elif intersection.geom_type == 'MultiPolygon':
        return list(intersection)
    else:
        raise IOError('Shape is not a Polygon or MultiPolygon.')


def construct_json_region(image_w, image_h, image_bbox, row):
    x_coords = np.array(row['Overlap'].exterior)[:, 0]
    y_coords = np.array(row['Overlap'].exterior)[:, 1]

    # Convert coordinates to image gridpoints
    minx, miny, maxx, maxy = image_bbox.bounds
    x_gridpoints = np.ceil((x_coords - minx) / (maxx - minx) * image_w).astype(int).tolist()
    y_gridpoints = np.ceil((y_coords - maxy) / (miny - maxy) * image_h).astype(int).tolist()

    return {
        'shape_attributes': {
            'name': "polygon",
            'all_points_x': x_gridpoints,
            'all_points_y': y_gridpoints
        },
        'region_attributes': {'label': row['First_Order_Name']}
    }


def create_json_label_file(MW_data, image_filepaths, out_filepath):
    """
    Args:
        image_filepaths: filepaths for satellite images
        out_filepath: where to save output JSON file

    Returns: None
    """
    out_json = {}

    for image_num, image_filepath in enumerate(image_filepaths):
        image_filename = ntpath.basename(image_filepath)
        print(f"Image number {image_num + 1}: {image_filename}")  # to track progress
        image = plt.imread(image_filepath)

        # Extract bbox from image filename
        processed_filename = image_filename[len('osm11_res10m'):]
        string_coordinates = re.findall(r'-?\d+\.?\d*', processed_filename)
        minx, miny, maxx, maxy = [float(s) for s in string_coordinates]
        bbox = box(minx, miny, maxx, maxy)

        # Find the polygons which overlap with the bbox
        overlapping_polygons = data2018.copy().loc[MW_data.apply(lambda row: check_for_overlap(bbox, row), axis=1)]
        # get_overlap(...) may return a list[Polygon] if there are multiple distinct areas of overlap between the bbox
        # and the Polygon. explode(...) creates a row for each of the Polygons in the list[Polygon]
        overlapping_polygons['Overlap'] = [get_overlap(bbox, plygn) for plygn in overlapping_polygons['Location']]
        overlapping_polygons = overlapping_polygons.explode('Overlap')

        # Construct information (key and value) for out_json dict
        file_size = os.path.getsize(image_filepath)
        out_json_key = image_filename + str(file_size)

        regions = []
        h, w, *_ = image.shape
        overlapping_polygons.apply(lambda row: regions.append(construct_json_region(w, h, bbox, row)), axis=1)
        out_json_value = {
            'filename': image_filename,
            'size': file_size,
            'regions': regions
        }

        # Save to image-entry to out_json dict
        if out_json_key in out_json.keys():
            print(f"Oops: {out_json_key} is already a key in the out_json dictionary")
        out_json[out_json_key] = out_json_value

    with open(out_filepath, 'w') as outfile:
        json.dump(out_json, outfile)


if __name__ == '__main__':
    MW_DATA_FILEPATH = os.path.join('data', 'lcdb-v50-land-cover-database-version-50-mainland-new-zealand.csv')
    import_start = timer()
    with open(MW_DATA_FILEPATH) as csvfile:
        MW_data = pd.read_csv(csvfile)
        MW_data.rename(columns={MW_data.columns[0]: 'Location'}, inplace=True)  # column name is gibberish otherwise
    import_end = timer()
    print(f"Time to import CSV file: {import_end - import_start}s")

    # Extract the 2018 information
    data2018 = MW_data.loc[:, ['Location', 'Name_2018', 'Class_2018', 'Wetland_18', 'Onshore_18', 'LCDB_UID']]
    data2018.columns = ['Location', 'Name', 'Class', 'Wetland', 'Onshore', 'ID']

    # Transform column of locations (string) into polygons
    transform_start = timer()
    data2018['Location'] = data2018['Location'].map(wkt.loads)
    transform_end = timer()
    print(f"Time to transform locations into Polygons: {transform_end - transform_start}s")

    # Construct dictionary where keys are class-names and values are the associated first-order class-names
    FIRST_ORDER_NAMES = {
        **dict.fromkeys(['Built-up Area (settlement)', 'Urban Parkland/Open Space', 'Surface Mine or Dump',
                         'Transport Infrastructure'],
                        'Artificial Surfaces'),
        **dict.fromkeys(['Sand or Gravel', 'Gravel or Rock', 'Landslide', 'Permanent Snow and Ice',
                         'Alpine Grass/Herbfield'],
                        'Bare or Lightly-vegetated Surfaces'),
        **dict.fromkeys(['Lake or Pond', 'River', 'Estuarine Open Water'],
                        'Water Bodies'),
        **dict.fromkeys(['Short-rotation Cropland', 'Orchard, Vineyard or Other Perennial Crop'],
                        'Cropland'),
        **dict.fromkeys(['High Producing Exotic Grassland', 'Low Producing Grassland', 'Tall Tussock Grassland',
                         'Depleted Grassland', 'Herbaceous Freshwater Vegetation', 'Herbaceous Saline Vegetation',
                         'Flaxland'],
                        'Grassland, Sedgeland and Marshland'),
        **dict.fromkeys(['Fernland', 'Gorse and/or Broom', 'Manuka and/or Kanuka', 'Matagouri or Grey Scrub',
                         'Broadleaved Indigenous Hardwoods', 'Sub Alpine Shrubland', 'Mixed Exotic Shrubland'],
                        'Scrub and Shrubland'),
        **dict.fromkeys(['Exotic Forest', 'Forest - Harvested', 'Deciduous Hardwoods', 'Indigenous Forest', 'Mangrove'],
                        'Forest')
    }
    # Remove polygons which don't have a class
    data2018 = data2018[data2018['Name'] != "Not land"]
    # Create column which contains the first order names
    data2018['First_Order_Name'] = data2018['Name'].map(FIRST_ORDER_NAMES)

    # Check for NaN values:
    data2018_nan = data2018[data2018.isna().any(axis=1)]
    if len(data2018_nan) > 0:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(data2018_nan)
        raise KeyError("There is keys missing from the first_order_names dictionary.")

    # WARNING: requires that the folder only contains appropriate JPEG files (no other files or folders)
    IMAGES_FOLDER = os.path.join('nz_images', 'osm11')
    image_filenames = [os.path.join(IMAGES_FOLDER, a) for a in os.listdir(IMAGES_FOLDER)]

    create_json_start = timer()
    create_json_label_file(data2018, image_filenames, os.path.join('nz_images', 'MW_annotations.json'))
    create_json_end = timer()
    print(f"Time to create JSON file: {create_json_end - create_json_start}s")

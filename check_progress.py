"""
Given an annotation/label JSON file, this counts the number of vertex points, instances, and images with at least one
instance.
"""

import json

FILEPATH = './nz_images/xavier_annotations2.json'

with open(FILEPATH) as json_file:
    annotations = json.load(json_file)

num_vertex_points, num_instances, num_images_annotated = 0, 0, 0
for image in annotations.values():
    num_instances += len(image['regions'])

    if len(image['regions']) > 0:
        for polygon in image['regions']:
            num_vertex_points += len(polygon['shape_attributes']['all_points_x'])

        num_images_annotated += 1

print(f"Number of vertex points:     {num_vertex_points}\n"
      f"Number of instances:         {num_instances}\n"
      f"Number of annotated images:  {num_images_annotated}")

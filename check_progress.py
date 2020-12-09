"""
Counts an image as "annotated" if it has at least one annotation
"""

import json

FILEPATH = './nz_images/xavier_annotations4.json'

with open(FILEPATH) as json_file:
    annotations = json.load(json_file)

num_vertex_points, num_instances, num_images_annotated = 0, 0, 0
for image in annotations.items():
    num_instances += len(image[1]['regions'])

    if len(image[1]['regions']) > 0:
        for polygon in image[1]['regions']:
            num_vertex_points += len(polygon['shape_attributes']['all_points_x'])

        num_images_annotated += 1

print(f'Number of vertex points:    {num_vertex_points}\n'
      f'Number of instances:        {num_instances}\n'
      f'Number of annotated images: {num_images_annotated}')

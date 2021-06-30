"""
Given an annotation/label JSON file, this counts the number of vertex points, instances, and images with at least one
instance.
"""

import os
import json

FILEPATH = os.path.join('nz_images', 'manual_annotations.json')

with open(FILEPATH) as json_file:
    annotations = json.load(json_file)

num_vertex_points, num_instances, num_images_annotated = 0, 0, 0
class_counts = dict()
for image in annotations.values():
    num_instances += len(image['regions'])

    if len(image['regions']) > 0:
        for polygon in image['regions']:
            num_vertex_points += len(polygon['shape_attributes']['all_points_x'])

            if "label" not in polygon['region_attributes'].keys():
                # Report unlabelled regions
                print(f"Unlabeled region in {image['filename']}")
            else:
                # Add one to count for that class
                class_name = polygon['region_attributes']['label']
                if class_name in class_counts.keys():
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

        num_images_annotated += 1

print(f"Number of vertex points:     {num_vertex_points}\n"
      f"Number of instances:         {num_instances}\n"
      f"Number of annotated images:  {num_images_annotated}\n")

print("Class-breakdown of instances")
for class_name, class_count in class_counts.items():
    print(f"{class_name}: {class_count}")

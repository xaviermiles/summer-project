"""
Given folder of images and associated JSON label file, this script constructs corresponding class images (where each
colour is associated with a class).
If a region has a class-label 'nan' in the JSON file, then it is recorded in the image as 99 and adds one to the
'white count'.
The JSON file should be in the current working directory
"""

import os
import json
from PIL import Image, ImageDraw


def create_class_images(original_json, out_folder_suffix, data_dir, converter):
    """
    Assumes: all annotations in original_json have a corresponding JPEG image in either data_dir/train or data_dir/val
    
    Requires:
    - original_json: JSON file containing all annotations
    - out_folder_suffix: string; identifier that will be attached to the train and val folders
    - data_dir: directory containing train and val true-colour images, class image folders will be made here
    - converter: dictionary[string -> int]; maps class names to integer
    """
    # Import annotations
    with open(original_json) as jsonfile:
        labels = json.load(jsonfile)

    # Set folders to save class images to
    train_class_dir = os.path.join(data_dir, 'train_class_images_' + out_folder_suffix)
    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
    val_class_dir = os.path.join(data_dir, 'val_class_images_' + out_folder_suffix)
    if not os.path.exists(val_class_dir):
        os.makedirs(val_class_dir)

    print(f"Processing: {original_json}...\n"
          f"Processing image number:", end="", flush=True)
    nan_count = 0
    for image_num, image_info in enumerate(labels.values()):
        # print progress
        if (image_num + 1) % 20 == 0:
            print(f"  {image_num + 1}", end="", flush=True)
        
        filename = image_info['filename']
        if filename in os.listdir(os.path.join(data_dir, 'train')):
            in_filepath = os.path.join(data_dir, 'train', filename)
            out_filepath = os.path.join(train_class_dir, filename)
        elif filename in os.listdir(os.path.join(data_dir, 'val')):
            in_filepath = os.path.join(data_dir, 'val', filename)
            out_filepath = os.path.join(val_class_dir, filename)

        # Get dimensions of original image
        image = Image.open(in_filepath)
        w, h = image.size

        # Construct class image
        class_image = Image.new(mode="P", size=(w, h), color=0)  # every pixel is initialised as black
        class_image1 = ImageDraw.Draw(class_image)
        for regions in image_info['regions']:
            polygon_info = regions['shape_attributes']
            xy = list(zip(polygon_info['all_points_x'], polygon_info['all_points_y']))

            class_name = regions['region_attributes']['label']
            if class_name in converter.keys():
                fill = converter[class_name]
            else:
                nan_count += 1

            class_image1.polygon(xy, fill=fill)

        # Save to disk (with same filename as original image)
        class_image.convert('RGB').save(out_filepath, "JPEG")

    # Print the number of nan entries
    print(f"\nNumber of NAN entries: {nan_count}")

    
if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "data_aug")
    
    manual_class_to_colours = {
        'water': 1,
        'vegetation': 2
    }
    create_class_images("manual_annotations_aug.json", "manual", data_dir, manual_class_to_colours)
    
    MW_class_to_colours = {
        'Artificial Surfaces': 1,
        'Bare or Lightly-vegetated Surfaces': 2,
        'Water Bodies': 3,
        'Cropland': 4,
        'Grassland, Sedgeland and Marshland': 5,
        'Scrub and Shrubland': 6,
        'Forest': 7
    }
#     create_class_images("MW_annotations_aug.json", "MW", data_dir, MW_class_to_colours)


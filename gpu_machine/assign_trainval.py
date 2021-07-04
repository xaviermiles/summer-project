"""
Some functions to help split a set of JPEG images and their associated annotations (JSON) into train and val sets.
"""

import json
import random
import os
import shutil


def copy_jpegfiles(original_image_dir, data_dir, val_jpegnames):
    """
    Copies JPEG files from folder into train and val folders.
    Creates dataset directory (and subdirectories) if they do no exist
    
    Requires:
    - original_image_dir: directory where original JPEG images are stored (WARNING: takes all
    JPEG files from folder)
    - data_dir: directory to put train and val folders into
    - val_jpegnames: list[string]; JPEG filenames which to put in val folder, all others will
    be put in train folder (WARNING: assumes val_jpegnames are files that exist in original_image_dir)
    
    """
    for direc in [os.path.join(data_dir, "train"), os.path.join(data_dir, "val")]:
        if not os.path.exists(direc):
            os.makedirs(direc)
    
    # recreate list of all JPEG filenames
    jpegnames = [file for file in os.listdir(ORIGINAL_IMAGES_DIR) if file.endswith('.jpeg')]
    
    print(f"Copying {len(jpegnames)} images...")
    # copy each JPEG according to whether it should be in train or val
    for fname in jpegnames:
        fpath = os.path.join(original_image_dir, fname)
        
        if fname in val_jpegnames:
            shutil.copy(fpath, os.path.join(data_dir, "val"))
        else:
            shutil.copy(fpath, os.path.join(data_dir, "train"))


def create_trainval_annotations(original_jsonfile, out_json_prefix, data_dir, val_jpegnames):
    """
    Creates dataset directory (and subdirectories) if they do no exist
    
    Requires:
    - original_jsonfile: JSON filename which contains annotations for the images
    - out_json_prefix: string; to distinguish multiple train/val JSON files 
    (eg "manual" -> manual_val.json, manual_train.json)
    - data_dir: directory to put train and val folders into
    - val_jpegnames: list[string]; JPEG filenames which to put in val annotations files, all others will
    be put in train annotations file (WARNING: assumes val_jpegnames are filenames present in original_jsonfile)
    """
    for direc in [os.path.join(data_dir, "train"), os.path.join(data_dir, "val")]:
        if not os.path.exists(direc):
            os.makedirs(direc)
    
    # retrieve complete annotations
    print(f"Processing: {original_jsonfile}")
    with open(original_jsonfile) as f:
        annotations = json.load(f)
        
    # split annotations into dictionaries
    val, train = {}, {}
    for key in annotations.keys():
        if annotations[key]['filename'] in val_jpegnames:
            val[key] = annotations[key]
        else:
            train[key] = annotations[key]
        
    # write dictionaries to JSON
    out_val_path = os.path.join(data_dir, "val", out_json_prefix + "_val.json")
    with open(out_val_path, 'w') as f:
        json.dump(val, f)
        
    out_train_path = os.path.join(data_dir, "train", out_json_prefix + "_train.json")
    with open(out_train_path, 'w') as f:
        json.dump(train, f)
        

if __name__ == "__main__":
    # assign folder to write data to
    DATA_DIR = os.path.join(os.getcwd(), "data_aug")

    # path to original (complete) set of JPEG images to use for train and val
    ORIGINAL_IMAGES_DIR = os.path.join(os.getcwd(), "osm11_aug")

    # randomly select val images
    image_fnames = [file for file in os.listdir(ORIGINAL_IMAGES_DIR) if file.endswith('.jpeg')]
    random.seed(22)
    val_fnames = random.sample(image_fnames, int(len(image_fnames)/5))

    # copy JPEG images to appropriate folders
    copy_jpegfiles(ORIGINAL_IMAGES_DIR, DATA_DIR, val_fnames)
    
    # create JSON annotations files corresponding to the train and val JPEG sets
    create_trainval_annotations("manual_annotations_aug.json", "manual", DATA_DIR, val_fnames)
#     create_trainval_annotations("MW_annotations.json", "MW", DATA_DIR, val_fnames)
    
    print("Finished")

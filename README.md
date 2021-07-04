# 2020/21 Summer project

This repo includes some of the code from a group project I worked on within the SAIL (Spatial And Image Learning) group at the University of Canterbury. 
The aim was to retrieve FREE satellite images of rural New Zealand and train neural networks (using image annotations/labelling) to learn different types of land cover (e.g. forest, water bodies, scrubland).
The report I produced for this project is included in the _other_ folder.

Overview:
- The satellite images were retrieved from [sentinelhub](https://www.sentinel-hub.com/) using their Python API (scripts: download_nz_images.py, image_utils.py). These images were constructed such that every part of New Zealand was included in at least one image. Each pixel corresponded to 10x10 physical metres as this was the highest resolution offered. Given this resolution and the New Zealand geojson shape, the splitting technique produced 1572 images necessary for downloading (the free trial was just sufficient).
- The original training annotations were manually generated by visually inspecting the satellite images and identifying regions of _vegetation_ and _water bodies_ (scripts:  check_progress.py, combine_json_files.py). (Including background pixels, this meant there was 3 classes).
- The neural networks which were trained/tested on the satellite images (Mask R-CNN, Deeplab) were run on a good-GPU computer not setup with git. Any program files copied from that "GPU machine" into this git repo were save in the folder _gpu_machine_, but the files in the folder don't make much sense without the context of the directory structure in the other computer (unfortunately).
- Late in the project, we were told about detailed land cover annotations maintained by Manaaki Whenua which encompass all of mainland New Zealand. I made a process to derive the appropriate annotations for each satellite image from the large collection of polygons in this land cover database (script: process_MW_data.py), but did not have time to test the neural network models with these annotations which included more classes ( and higher-detail polygons.

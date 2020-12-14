"""
Read the contents of multiple JSON files, combine these into one Python dictionary, and write to file.
"""

import json

json_filepaths = ['./nz_images/xavier_annotations.json', './nz_images/linda_annotations.json',
                  './nz_images/shinu_annotations.json']
json_contents = []
for json_filepath in json_filepaths:
    with open(json_filepath) as infile:
        json_contents.append(json.load(infile))

combined_json_contents = {}
for j in json_contents:
    for key, value in j.items():
        combined_json_contents[key] = value

# Write out JSON file
with open('./nz_images/combined_annotations.json', 'w') as outfile:
    json.dump(combined_json_contents, outfile)

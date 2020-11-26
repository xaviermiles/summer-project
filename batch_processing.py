"""
'large_area_utilities' showed how to split a large area into smaller bounding box which can then be obtained using the
requests demonstrated in 'processing_api_request'

Sentinel Hub Batch Processing shows another way to do that:
- takes the geometry of a large area and divides it according to a specified tile grid
- executes processing requests for each tile in the grid and stores results in a given location at AWS S3 storage
Because of the optimised performance, this is significantly faster than running the same process locally

1. Define and create a batch request
2. Analyse a batch request before it is executed
3. Run a batch requests job and check the outcome
"""

import os
import datetime as dt
import geopandas as gpd

from sentinelhub import SentinelHubBatch, SentinelHubRequest, Geometry, CRS, DataCollection, MimeType, SHConfig, \
    bbox_to_dimensions


# 1. Create a batch request
# 1.1 Define a Processing API request
CLIENT_ID = '9d542985-ed95-4357-b740-37596b1d5f70'
CLIENT_SECRET = ')(/q5|#Q[[C:Oy3Is46,kpQYR28ovX9y2@,6UP@D'
config = SHConfig()
if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

SHAPE_PATH = os.path.join('.', 'data', 'california_crop_fields.geojson')
"""
The main purpose of Feature Info Service (FIS) is to enable users to obtain statistics about satellite data without
actually having to download large amounts of raster imagery. This is most effective when you have a sparse collection of
bbox/polygons and would like to know aggregated values for each of them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import Polygon

from sentinelhub import FisRequest, BBox, Geometry, CRS, WcsRequest, CustomUrlParam, DataCollection, HistogramType
from sentinelhub.time_utils import iso_to_datetime
from sentinelhub import SHConfig

INSTANCE_ID = '2d26c335-1184-42e3-8190-5de92f6c6aac'
if INSTANCE_ID:
    config = SHConfig()
    config.instance_id = INSTANCE_ID
else:
    config = None


"""    
Exploring basic statistics
Example is concerned with Sahara desert during a sandstorm in March 2018
"""
sahara_bbox = BBox((1.0, 26.0, 1.3, 25.7), CRS.WGS84)
time_interval = ('2018-02-01', '2018-05-01')

fis_request = FisRequest(
    data_collection=DataCollection.SENTINEL2_L1C,
    layer='BANDS-S2-L1C',
    geometry_list=[sahara_bbox],
    time=time_interval,
    resolution='60m',
    data_folder='./data',
    config=config
)

# the call concurrently makes calls to FIS for each geometry and collects the responses
# save_data=True to avoid redownloading (~30s)
fis_data = fis_request.get_data(save_data=True)

print(fis_data[0]['C0'][0])


def fis_data_to_df(fis_data):
    """Creates a pd.DataFrame from list of FIS responses"""
    COLUMNS = ['channel', 'date', 'min', 'max', 'mean', 'stDev']
    data = []

    for fis_response in fis_data:
        for channel, channel_stats in fis_response.items():
            for stat in channel_stats:
                row = [int(channel[1:]), iso_to_datetime(stat['date'])]

                for column in COLUMNS[2:]:
                    row.append(stat['basicStats'][column])

                data.append(row)

    return pd.DataFrame(data, columns=COLUMNS).sort_values(['channel', 'date'])

df = fis_data_to_df(fis_data)
print(df)

# plot timeseries of mean values with standard deviation for each band
BANDS = 'B01,B02,B03,B04,B05,B06,B07,B08,B09,B10,B11,B12'.split(',')

plt.figure(figsize=(12, 8))
for channel, (band, color) in enumerate(zip(BANDS, cm.jet(np.linspace(0, 1, 13)))):
    channel_df = df[df.channel == channel]
    plt.plot(channel_df.date, channel_df['mean'], '-o', markeredgewidth=1,
             color=color, markeredgecolor='None', label=band)
    plt.fill_between(list(channel_df.date), channel_df['mean'] - channel_df['stDev'],
                     channel_df['mean'] + channel_df['stDev'], alpha=0.2, color=color)

plt.legend(loc='upper right')

"""
As there is no vegetation in the desert the reflectance values shouldn't change much over time. 
...(read details in tutorial doc)...
We can conclude the 5th,9th,14th acquisition contains clouds while 10th acquisition contains the sandstorm.
Let's download the true color images for these acquisitions and visually verify the results.
"""
wcs_request = WcsRequest(
    data_collection=DataCollection.SENTINEL2_L1C,
    layer='TRUE-COLOR-S2-L1C',
    bbox=sahara_bbox,
    time=time_interval,
    resx='60m',
    resy='60m',
    custom_url_params={CustomUrlParam.SHOWLOGO: False},
    config=config
)

images = wcs_request.get_data()

fig, axs = plt.subplots((len(images) + 2) // 3, 3, figsize=(8, 16))
for idx, (image, time) in enumerate(zip(images, wcs_request.get_dates())):
    axs.flat[idx].imshow(image)
    axs.flat[idx].set_title(time.date().isoformat())
fig.tight_layout()


"""
Comparing Histograms: FIS can also provide histograms of calues split into a specified number of binds which will allows
us to analyse the distribution of values without having to download entire images.

Compare NDVI value distributions of the following bboxes and polygons. We will divide values into 20 equally-sized bins
and use Landsat 8 data. For simplification, we select a time interval for which there is only one acquisition available.
"""
bbox1 = BBox([46.16, -16.15, 46.51, -15.58], CRS.WGS84)
bbox2 = BBox((1292344.0, 5195920.0, 1310615.0, 5214191.0), CRS.POP_WEB)

geometry1 = Geometry(Polygon([(-5.13, 48),
                              (-5.23, 48.09),
                              (-5.13, 48.17),
                              (-5.03, 48.08),
                              (-5.13, 48)]),
                     CRS.WGS84)
geometry2 = Geometry(Polygon([(1292344.0, 5205055.5),
                              (1301479.5, 5195920.0),
                              (1310615.0, 5205055.5),
                              (1301479.5, 5214191.0),
                              (1292344.0, 5205055.5)]),
                     CRS.POP_WEB)

print(list(HistogramType))

ndvi_script = 'return [(B05 - B04) / (B05 + B04)]'

histogram_request = FisRequest(
    data_collection=DataCollection.LANDSAT8,
    layer='TRUE-COLOR-L8',
    geometry_list=[bbox1, bbox2, geometry1, geometry2],
    time=('2018-06-10', '2018-06-15'),
    resolution='100m',
    bins=20,
    histogram_type=HistogramType.EQUIDISTANT,
    custom_url_params={CustomUrlParam.EVALSCRIPT: ndvi_script},
    config=config
)

histogram_data = histogram_request.get_data() # get a FIS response for each of the 4 areas

print(histogram_data[0])

# plotting histogram data
plot_data = []
for idx, fis_response in enumerate(histogram_data):
    bins = fis_response['C0'][0]['histogram']['bins']

    counts = [value['count'] for value in bins]
    total_counts = sum(counts)
    counts_perc = [round(100 * count / total_counts) for count in counts] # scaled to percentage of total

    bin_size = bins[1]['lowEdge'] - bins[2]['lowEdge'] # as the bins are equally-sized
    splits = [value['lowEdge'] + bin_size / 2 for value in bins] # calculate middle of bins

    data = []
    for count, split in zip(counts_perc, splits):
        data.extend([split] * count)

    plot_data.append(np.array(data))

print(plot_data)

# CANNOT INSTALL SEABORN
# fig, ax = plt.subplots(figsize=(12, 8))
# ax = sns.violinplot(plot_data, )
# ax.set(xticklabels=[f'Area {idx}' for idx in range(len(histogram_data))], ylabel='NDVI', fontsize=15)

plt.show()
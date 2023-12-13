# %%
from osgeo import gdal
import warnings
from pathlib import Path
import os
from datetime import datetime
from functools import partial
from multiprocessing.pool import ThreadPool

import rasterio
from rasterio import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import numpy as np
import geopandas as gpd
import ee
from pyproj import CRS

from gdstools import (
    GEEImageLoader,
    ConfigLoader,
    infer_utm,
    multithreaded_execution,
    split_bbox,
    print_message
)

# warnings.filterwarnings("ignore", message="Default upsampling behavior")
# warnings.filterwarnings(
#     'ignore', category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings(
    'ignore', category=rasterio.errors.RasterioDeprecationWarning)
# Get cut-down GDAL that rasterio uses
# ... and suppress errors
gdal.PushErrorHandler('CPLQuietErrorHandler')


def maskS2clouds(image):
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)\
             .And(qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)


# %%
def get_sentinel2(
    bbox,
    year,
    path=None,
    as_array=False,
    prefix=None,
    season="leafon",
    bands=None,
    gee_collection='COPERNICUS/S2_HARMONIZED',
    alias='Sentinel2HAR',
    overwrite=False,
    epsg=4326,
    scale=10,
):
    """
    Fetch Sentinel 2 image url from Google Earth Engine (GEE) using a bounding box.

    Catalog https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED

    Parameters
    ----------
    month : int
        Month of year (1-12)
    year : int
        Year (e.g. 2019)
    bbox : list
        Bounding box in the form [xmin, ymin, xmax, ymax].

    Returns
    -------
    url : str
        GEE generated URL from which the raster will be downloaded.
    metadata : dict
        Image metadata.
    """
    if season == "leafoff":
        start_date = f"{year - 1}-10-01"
        end_date = f"{year}-03-31"
    elif season == "leafon":
        start_date = f"{year}-04-01"
        end_date = f"{year}-09-30"
    else:
        raise ValueError(f"Invalid season: {season}")

    bbox = ee.Geometry.BBox(*bbox)
    collection = ee.ImageCollection(gee_collection)\
                   .filterDate(start_date, end_date)\
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
                   .map(maskS2clouds)\
                   .filterBounds(bbox)\
                   .select(bands)

    ts_start = datetime.timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
    ts_end = datetime.timestamp(datetime.strptime(end_date, "%Y-%m-%d"))

    # Download forest layer only
    # Reclass probabilities to 0-1
    # img = collection.select(BANDS)
    img_median = collection.median()

    # Set image metadata and params
    def get_medoid(image):
        diff = ee.Image(image).subtract(img_median).pow(ee.Image.constant(2))
        return diff.reduce('sum').addBands(image)

    medoid = collection.map(get_medoid)\
                       .reduce(ee.Reducer.min(len(bands)))\
                       .select([x for x in range(1, len(bands) + 1)], bands)\
                       .clip(bbox)

    # image = GEEImageLoader(medoid)
    image = GEEImageLoader(img_median)
    image.metadata_from_collection(collection)
    image.set_property("system:time_start", ts_start * 1000)
    image.set_property("system:time_end", ts_end * 1000)
    image.set_params("scale", scale)
    image.set_params("crs", f"EPSG:{epsg}")
    image.set_params("region", bbox)
    image.set_viz_params("min", 0)
    image.set_viz_params("max", 0.3)
    image.set_viz_params('bands', ['B12', 'B8A', 'B4'])
    image.id = f"{prefix}{year}_{alias}_{season}"

    # Download cog
    # out_path = path / image.id
    # out_path.mkdir(parents=True, exist_ok=True)

    if as_array:
        return image.to_array()
    else:
        # image.save_metadata(path)
        # image.save_preview(out_path, overwrite=overwrite)
        image.to_geotif(path, overwrite=overwrite)


def quad_fetch(
    bbox,
    path,
    dim=1,
    num_threads=None,
    overwrite=False,
    progressbar=None,
    **kwargs
):
    """Breaks user-provided bounding box into quadrants and retrieves data
    using `fetcher` for each quadrant in parallel using a ThreadPool.

    Parameters
    ----------
    fetcher : callable
      data-fetching function, expected to return an array-like object
    bbox : 4-tuple or list
      coordinates of x_min, y_min, x_max, and y_max for bounding box of tile
    num_threads : int
      number of threads to use for parallel executing of data requests
    qq : bool
      whether or not to execute request for quarter quads, which executes this
      function recursively for each quadrant
    *args
      additional positional arguments that will be passed to `fetcher`
    **kwargs
      additional keyword arguments that will be passed to `fetcher`

    Returns
    -------
    quad_img : array
      image returned with quads stitched together into a single array

    """
    filename = f"{kwargs['prefix']}{kwargs['state']}_{kwargs['year']}_{kwargs['alias']}_{kwargs['season']}-cog"
    filename = path / f"{filename}.tif"

    # Remove kwargs not used by get_sentinel2
    del kwargs["state"]

    if os.path.exists(filename) and overwrite is False:
        msg = f"File already exists: {filename}. Set overwrite to True to download it again."
        print_message(msg, progressbar)
        return

    if dim > 1:
        if num_threads is None:
            num_threads = dim**2

        bboxes = split_bbox(dim, bbox)
        n = len(bboxes)

        get_quads = partial(get_sentinel2, **kwargs)
        with ThreadPool(num_threads) as p:
            quads = p.map(get_quads, bboxes)

        # Split quads list in tuples of size dim
        quad_zip = list(zip(*quads))
        quad_list = [list(quad_zip[0][x:x + dim])
                     for x in range(0, len(quad_zip[0]), dim)]
        # Reverse order of rows to match rasterio's convention
        [x.reverse() for x in quad_list]
        mosaic = np.dstack([np.concatenate(quad_list[x], 1)
                           for x in range(0, len(quad_list))])

        profiles = [list(quad_zip[1][x:x + dim])
                    for x in range(0, len(quad_zip[1]), dim)]
        PROFILE = profiles[0][-1]  # profile from first quad (left upper)
        PROFILE.update(
            width=mosaic.shape[2], height=mosaic.shape[1], dtype=rasterio.float32)
        cog_profile = cog_profiles.get("deflate")

        with MemoryFile() as memfile:
            with memfile.open(**PROFILE) as dst:
                dst.write(mosaic)

                cog_translate(
                    dst,
                    filename,
                    cog_profile,
                    in_memory=True,
                    quiet=True
                )

    else:
        dem = get_sentinel2(bbox, **kwargs)
        return dem


def bbox_padding(geom, padding=1e3):
    p_crs = infer_utm(geom.bounds)
    p_geom = gpd.GeoSeries(geom, crs=4326).to_crs(p_crs)
    if padding > 0:
        p_geom = p_geom.buffer(padding, join_style=2)

    return p_geom.to_crs(4326).bounds.values[0]


# %%
if __name__ == "__main__":

    # Load configuration parameters
    YEAR = 2022
    run_as = 'dev'

    conf = ConfigLoader(Path(__file__).parent.parent).load()
    datadir = Path(conf.DATADIR) / 'tiles'
    grid = gpd.read_file(conf.GRID)
    if run_as == 'dev':
        datadir = Path(conf.DEV_DATADIR) / 'tiles'
        grid = grid.sort_values('CELL_ID').iloc[:10]

    datarepo = Path(conf.DATADIR) / 'interim'
    alias = conf.sentinel2sr['alias']
    api_url = conf.sentinel2sr['api_url']
    bands = list(conf.sentinel2sr['bands'].keys())
    collection = conf.sentinel2sr['collection']

    # Initialize the Earth Engine module.
    # Setup your Google Earth Engine API key before running this script.
    # Instructions available at https://cloud.google.com/sdk/docs/install#deb
    # %%
    # ee.Authenticate() # run once after installing gcloud api
    ee.Initialize(opt_url=api_url)

    dw_path = datadir / alias.lower() / str(YEAR)
    dw_path.mkdir(parents=True, exist_ok=True)

    # %%
    # Select a subset of qq cells to play with.
    # qq_shp = qq_shp[qq_shp.CELL_ID.isin(qq_shp.head(20).CELL_ID)].copy()
    params = [
        {
            'bbox': row.geometry.bounds, #bbox_padding(row.geometry),
            'dim': 2,
            'year': YEAR,
            'state': row.STATE,
            'path': dw_path,
            'bands': bands,
            'alias': alias,
            'as_array': True,
            'prefix': f'{row.CELL_ID}_',
            'season': "leafon",
            'overwrite': False
        } for row in grid.itertuples()
    ]

    # quad_fetch(**params[0])
    multithreaded_execution(quad_fetch, params, 20)

    # params[params == 'leafon'] = 'leafoff'
    # multithreaded_download(params, get_dworld)

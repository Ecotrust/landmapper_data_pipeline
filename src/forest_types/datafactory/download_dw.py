# %%
from pathlib import Path
import glob
from datetime import datetime

import numpy as np
import geopandas as gpd
import ee
from pyproj import CRS

from gdstools import multithreaded_execution, GEEImageLoader, ConfigLoader


# %%
def get_dworld(
    bbox,
    year,
    path,
    prefix=None,
    season="leafon",
    overwrite=False,
    epsg=4326,
    scale=10,
    progressbar=None,
):
    """
    Fetch Dynamic World image url from Google Earth Engine (GEE) using a bounding box.

    Catalog https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1

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

    def countPixels(image, geometry):
        counts = image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e13,
        )
        try:
            return counts.get("label").getInfo()
        except:
            return 0

    if season == "leafoff":
        start_date = f"{year - 1}-10-01"
        end_date = f"{year}-03-31"
    elif season == "leafon":
        start_date = f"{year}-04-01"
        end_date = f"{year}-09-30"
    else:
        raise ValueError(f"Invalid season: {season}")

    bbox = ee.Geometry.BBox(*bbox)
    collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start_date, end_date)
        .filterBounds(bbox)
        .select("label")
    )

    ts_start = datetime.timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
    ts_end = datetime.timestamp(datetime.strptime(end_date, "%Y-%m-%d"))

    # Download forest layer only
    # Extract medoid from collection
    # Reclass probabilities to 0-1
    img_median = collection.reduce(ee.Reducer.median())

    def get_medoid(image):
        diff = ee.Image(image).subtract(img_median).pow(ee.Image.constant(2))
        return diff.reduce("sum").addBands(image)

    medoid = (
        collection.map(get_medoid)
        .reduce(ee.Reducer.min(2))
        .select([1], ["label"])
        .clip(bbox)
    )
    #    .eq(1)\
    # reclass = medoid.expression('b(0) == 1 ? 1 : 0')
    # tree_pixs = countPixels(medoid.mask(reclass), bbox)
    # all_pixs = countPixels(medoid, bbox)

    # Skip if forest pixels are less than 10% of total pixels
    # Handle division by zero error
    # try:
    #     pct = tree_pixs / all_pixs
    # except:
    #     pct = 0

    # if pct >= 0.1:
    image = GEEImageLoader(img_median)
    # Set image metadata and params
    # image.metadata_from_collection(collection)
    image.set_property("system:time_start", ts_start * 1000)
    image.set_property("system:time_end", ts_end * 1000)
    image.set_params("scale", scale)
    image.set_params("crs", f"EPSG:{epsg}")
    image.set_params("region", bbox)
    # image.set_viz_params("min", 0)
    # image.set_viz_params("max", 1)
    # image.set_viz_params('palette', ['black', '397D49'])
    image.id = f"{prefix}{year}_DynamicWorld_{season}"

    # Download cog
    # out_path = path / image.id
    # out_path.mkdir(parents=True, exist_ok=True)

    # image.save_metadata(out_path)
    image.to_geotif(path, overwrite=overwrite)
    # image.save_preview(out_path, overwrite=True)
    # else:
    #     print(f"Tile {prefix.replace('_', '')} has less than 10% of forest pixels, skipping...")


def infer_utm(bbox):
    """Infer the UTM Coordinate Reference System (CRS) by determining
    the UTM zone where a given lat/long bounding box is located.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)

    Returns
    -------
    crs : pyproj.CRS
      UTM crs for the bounding box
    """
    xmin, _, xmax, _ = bbox
    midpoint = (xmax - xmin) / 2

    if xmax <= -120 + midpoint:
        epsg = 32610
    elif (xmin + midpoint > -120) and (xmax <= -114 + midpoint):
        epsg = 32611
    elif (xmin + midpoint > -114) and (xmax <= -108 + midpoint):
        epsg = 32612
    elif (xmin + midpoint > -108) and (xmax <= -102 + midpoint):
        epsg = 32613
    elif (xmin + midpoint > -102) and (xmax <= -96 + midpoint):
        epsg = 32614
    elif (xmin + midpoint > -96) and (xmax <= -90 + midpoint):
        epsg = 32615
    elif (xmin + midpoint > -90) and (xmax <= -84 + midpoint):
        epsg = 32616
    elif (xmin + midpoint > -84) and (xmax <= -78 + midpoint):
        epsg = 32617
    elif (xmin + midpoint > -78) and (xmax <= -72 + midpoint):
        epsg = 32618
    elif xmin + midpoint > -72:
        epsg = 32619

    return CRS.from_epsg(epsg)


def bbox_padding(geom, padding=1e3):
    p_crs = infer_utm(geom.bounds)
    p_geom = gpd.GeoSeries(geom, crs=4326).to_crs(p_crs)
    if padding > 0:
        p_geom = p_geom.buffer(padding, join_style=2)

    return p_geom.to_crs(4326).bounds.values[0]


# %%
if __name__ == "__main__":
    # %%
    # Load config
    run_as = 'dev'
    YEAR = 2023
    conf = ConfigLoader(Path(__file__).parent.parent).load()
    dw = conf.dynamic_world
    WORKERS = 20
    DATADIR = Path(conf.DATADIR) / "tiles"
    grid = gpd.read_file(conf.GRID)
    if run_as == 'dev':
        DATADIR = Path(conf.DEV_DATADIR) / "tiles"
        grid = grid.sort_values("CELL_ID").iloc[:10]

    # Initialize the Earth Engine module.
    # Setup your Google Earth Engine API key before running this script.
    # Instructions available at https://cloud.google.com/sdk/docs/install#deb
    # ee.Authenticate() # run once after installing gcloud api
    ee.Initialize(opt_url=dw["api_url"])

    # %%
    params = [
        {
            "bbox": row.geometry.buffer(0.004, join_style=2).bounds,
            "year": YEAR,
            "path": DATADIR / f"dynamic_world/{YEAR}",
            "prefix": f"{row.CELL_ID}_",
            "season": "leafon",
            "overwrite": True,
        }
        for row in grid.itertuples()
    ]

    # get_dworld(**params[0])
    multithreaded_execution(get_dworld, params, WORKERS)

    # params[params == 'leafon'] = 'leafoff'
    # multithreaded_download(params, get_dworld)

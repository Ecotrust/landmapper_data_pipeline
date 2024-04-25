import os
import sys
import argparse
from pathlib import Path

import ee
import geopandas as gpd

from gdstools import (
    multithreaded_execution, 
    save_cog
)

from src.utils.fetch import (
    landtrendr_from_gee,
)

from src import config

def download_images(
    row,
    out_dir,
    epsg=2992,
    year=2023,
    scale=30,
    overwrite=False,
):
    from rasterio.warp import calculate_default_transform
    cell_id = row.CELL_ID
    bbox = row.geometry.bounds
    # state = row.STATE

    # get landtrendr image and save to disk
    template = os.path.join(out_dir, f"sentinel2sr/{YEAR}", f"{cell_id}_{YEAR}_Sentinel2SR_leafon-cog.tif")
    out_lt = Path(os.path.join(out_dir, f"landtrendr/{year}", f"{cell_id}_{year}_LandTrendr-cog.tif"))

    if os.path.exists(out_lt) and not overwrite:
        return cell_id
    out_lt.parent.mkdir(exist_ok=True, parents=True)
    
    try:
        lt_ras, lt_profile = landtrendr_from_gee(bbox, year, epsg, scale=scale)
        save_cog(lt_ras, lt_profile, out_lt, overwrite=overwrite)

    except Exception as e:
        print("Failed landtrendr on", cell_id, e)

    return cell_id


if __name__ == "__main__":
    # Load configuration parameters
    # config = ConfigLoader(Path(__file__).parent.parent).load()
    parser = argparse.ArgumentParser('Fetch LandTrendr data from Google Earth Engine.')
    parser.add_argument('-y', '--year', help='Year to download', type=int)
    parser.add_argument('--dev', help='Run scrip in dev mode', action="store_false")
    parser.add_argument('-o', '--overwrite', help='Overwrite file if already exists', action="store_true")
    args = parser.parse_args(sys.argv[2:])

    # Load configuration parameters
    YEAR = args.year

    datadir = Path(config.DATADIR) / 'processed/tiles' 
    grid = gpd.read_file(config.GRID)
    if args.dev:
        # datadir = Path(conf.DEV_DATADIR) / 'processed/tiles' 
        cellids = [273467, 184829, 134775, 247655, 116879, 132833, 213207, 181485, 132835, 116877, 181694, 114676, 298270, 228075, 184595, 312284, 144231, 270215, 275909]
        # 2023 data for these tiles failed to download. Fetching 2022 data instead
        # cellids = [176291, 198598, 198600, 305390, 151610, 123960, 144229, 222119, 154101]
        # 2022 data for these tiles failed to download. Fetching 2021 data instead
        # cellids = [144229]
        grid = grid[grid.CELL_ID.isin(cellids)]
        # grid = grid.sort_values('CELL_ID').iloc[:10]

    ee.Initialize()

    # Load the data
    params = [
        {
            "row": row,
            "out_dir": datadir,
            "epsg": grid.crs.to_epsg(),
            "year": YEAR,
            "scale": 30,
            "overwrite": True,
        } for row in grid.itertuples()
    ]

    # Download the data
    multithreaded_execution(download_images, params, threads=8)
    # download_images(**params[0])

    print("Done!")

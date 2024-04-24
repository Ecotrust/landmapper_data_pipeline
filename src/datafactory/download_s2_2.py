# %%
import os
from pathlib import Path

import ee
import geopandas as gpd
import rasterio
from tqdm.notebook import tqdm

# %%
from gdstools import (
    multithreaded_execution
)
from src.utils.fetch import (
    s2_from_gee,
)
from src import config

ee.Initialize()

# %%
def download_images(
    gdf_row,
    out_dir,
    year=2023,
    epsg=4326,
    overwrite=False,
):
    bbox = gdf_row.geometry.bounds
    cell_id = gdf_row["CELL_ID"]

    # get sentinel image and save to disk
    out_s2 = os.path.join(out_dir, f"{cell_id}_{year}_Sentinel2SRH_leafon-cog.tif")
    try:
        if not os.path.exists(out_s2) or overwrite:
            s2_ras, s2_profile = s2_from_gee(bbox, year, epsg, scale=10)
            s2_profile.update(compress="LZW")

            with rasterio.open(out_s2, "w", **s2_profile) as dst:
                for i, band in enumerate(s2_ras):
                    dst.write(band, i + 1)

    except Exception as e:
        print("Failed sentinel on", cell_id, e)
        return cell_id

    return cell_id

# %%
tiles = gpd.read_file(config.GRID)
# tiles.to_crs(epsg=2992, inplace=True)
to_run = [row for _, row in tiles[["geometry", "CELL_ID"]].iterrows()]# if row.CELL_ID in [221801, 221802, 221804, 209974, 209976, 234342, 234344, 234359, 208526, 208527]]

# %%
OUT_DIR = Path(config.DATADIR) / "processed/tiles/sentinel2srh"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(os.path.join(OUT_DIR, "sentinel"), exist_ok=True)

params = [
    {
        "gdf_row": row,
        "out_dir": OUT_DIR,
        "year": 2023,
        "epsg": 4326,
        "overwrite": False,
    } for row in to_run
]

# download_images(**params[0])
multithreaded_execution(download_images, params)

# %%
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_raster
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import pickle

from gdstools import image_collection, multithreaded_execution  # , save_cog
from src.utils.load import load_sentinel, load_landtrendr, load_dem
from src import config

# %%
BAND_DESCRIPTIONS = {
    1: "Canopy Cover",
    2: "Dominant Height",
    3: "Live Tree Basal Area",
    4: "Quadratic Mean Diameter",
    5: "Trees Per Acre (Growing Stock)",
    6: "Aboveground Live Tree Biomass",
    7: "Dynamic World Land Cover",
}
BAND_UNITS = {
    1: "Percent",
    2: "Feet",
    3: "Square feet per acre",
    4: "Inches",
    5: "Trees Per Acre",
    6: "",
    7: "",
}


def load_features(
    s2_path,
    ltr_path,
    dem_path,
    dw_path,
    model,
    bounds=None,
    # to_crs=None,
    season="leafon",
):
    """Loads data from disk into a dataframe ready for predictive modeling.

    Parameters
    ----------
    path_to_rasters : str
      path to the directory where subdirectories for 'sentinel', 'landtrendr',
      and 'dem' imagery can be found.
    cell_id : int or str
      cell id which identifies a quarter quad.
    year : int or str
      year of imagery to select

    Returns
    -------
    df : DataFrame
      dataframe with feature data ready for predictive modeling
    """
    if season == "leafon":
        COL_ORDER = [
            "S2_R_LEAFON",
            "S2_G_LEAFON",
            "S2_B_LEAFON",
            "S2_NIR_LEAFON",
            "S2_SWIR1_LEAFON",
            "S2_SWIR2_LEAFON",
            "S2_RE1_LEAFON",
            "S2_RE2_LEAFON",
            "S2_RE3_LEAFON",
            "S2_RE4_LEAFON",
            "S2_NDVI_LEAFON",
            "S2_SAVI_LEAFON",
            "S2_BRIGHTNESS_LEAFON",
            "S2_GREENNESS_LEAFON",
            "S2_WETNESS_LEAFON",
            "LT_DUR_NBR",
            "LT_DUR_SWIR1",
            "LT_MAG_NBR",
            "LT_MAG_SWIR1",
            "LT_RATE_NBR",
            "LT_RATE_SWIR1",
            "LT_YSD_NBR",
            "LT_YSD_SWIR1",
            "ELEVATION",
            "LAT",
            "LON",
        ]
    else:
        COL_ORDER = [
            "S2_R_LEAFOFF",
            "S2_G_LEAFOFF",
            "S2_B_LEAFOFF",
            "S2_NIR_LEAFOFF",
            "S2_SWIR1_LEAFOFF",
            "S2_SWIR2_LEAFOFF",
            "S2_RE1_LEAFOFF",
            "S2_RE2_LEAFOFF",
            "S2_RE3_LEAFOFF",
            "S2_RE4_LEAFOFF",
            "S2_R_LEAFON",
            "S2_G_LEAFON",
            "S2_B_LEAFON",
            "S2_NIR_LEAFON",
            "S2_SWIR1_LEAFON",
            "S2_SWIR2_LEAFON",
            "S2_RE1_LEAFON",
            "S2_RE2_LEAFON",
            "S2_RE3_LEAFON",
            "S2_RE4_LEAFON",
            "S2_NDVI_LEAFON",
            "S2_SAVI_LEAFON",
            "S2_BRIGHTNESS_LEAFON",
            "S2_GREENNESS_LEAFON",
            "S2_WETNESS_LEAFON",
            "S2_NDVI_LEAFOFF",
            "S2_SAVI_LEAFOFF",
            "S2_BRIGHTNESS_LEAFOFF",
            "S2_GREENNESS_LEAFOFF",
            "S2_WETNESS_LEAFOFF",
            "S2_dR",
            "S2_dG",
            "S2_dB",
            "S2_dNIR",
            "S2_dSWIR1",
            "S2_dSWIR2",
            "S2_dRE1",
            "S2_dRE2",
            "S2_dNDVI",
            "S2_dSAVI",
            "S2_dBRIGHTNESS",
            "S2_dGREENNESS",
            "S2_dWETNESS",
            "S2_dRE3",
            "S2_dRE4",
            "LT_DUR_NBR",
            "LT_DUR_SWIR1",
            "LT_MAG_NBR",
            "LT_MAG_SWIR1",
            "LT_RATE_NBR",
            "LT_RATE_SWIR1",
            "LT_YSD_NBR",
            "LT_YSD_SWIR1",
            "ELEVATION",
            "LAT",
            "LON",
        ]

    # if to_crs:
    #     s2_a, profile = reproject_raster(s2_path, to_crs, bounds=bounds, resolution=10)
    #     ltr_a, _ = reproject_raster(ltr_path, to_crs, bounds=bounds, resolution=10)
    #     dem_a, dem_profile = reproject_raster(
    #         dem_path, to_crs, bounds=bounds, resolution=10
    #     )
    #     dw_a, _ = reproject_raster(dw_path, to_crs, bounds=bounds, resolution=10)
    #     height, width = profile["height"], profile["width"]

    #     s2 = load_sentinel(s2_a, season=season)
    #     lt = load_landtrendr(ltr_a)
    #     dem = load_dem(dem_a[0,], dem_profile)
    s2 = load_sentinel(s2_path, bounds=bounds, season=season)
    lt = load_landtrendr(ltr_path, bounds=bounds, height=696, width=696)
    dem = load_dem(dem_path, bounds=bounds)

    with rasterio.open(dw_path) as src:
        profile = src.profile
        if bounds:
            window = src.window(*bounds)
            height, width = round(window.height), round(window.width)
            profile.update(
                height=height, width=width, transform=src.window_transform(window)
            )
            dw_a = src.read(window=window)
        else:
            height, width = src.height, src.width
            dw_a = src.read()

    profile.update(count=7, dtype="int16", nodata=-9999)

    assert s2.shape[0] == lt.shape[0] == dem.shape[0], "Shape mismatch"

    X = pd.concat([s2, lt, dem], axis=1)[COL_ORDER]

    # check for nodata in input dataframe
    pred = model.predict(X)
    for i in range(pred.shape[1]):
        if i == 0:
            pred[:, i] = np.clip(pred[:, i], 0, 100)  # clip canopy cover to 0-100
        else:
            pred[:, i] = np.clip(pred[:, i], 0, None)  # clip negative structure values
    pred = pred.reshape((height, width, 6))

    # with rasterio.open(dw_path) as dw_src:
    #     dw = dw_src.read(1, window=window)

    pred[:][~np.isin(dw_a[0,], [1, 5])] = profile["nodata"]

    pred_ras = reshape_as_raster(pred).astype("int16")

    img = np.vstack((pred_ras, dw_a.reshape(1, height, width)))

    return img, profile


def load_predict_save(
    s2_path, ltr_path, dem_path, dw_path, outfile, model, bounds=None, overwrite=False
):
    cell_id = Path(s2_path).stem.split("_")[0]
    to_crs = None  # CRS.from_epsg(2992)
    if not os.path.exists(outfile) or overwrite:
        try:
            pred_ras, profile = load_features(
                s2_path,
                ltr_path,
                dem_path,
                dw_path,
                model,
                bounds=bounds,
                to_crs=to_crs,
            )

            with rasterio.open(outfile, "w", **profile) as dst:
                for i in range(profile["count"]):
                    dst.write(pred_ras[i], i + 1)
                    dst.set_band_description(i + 1, BAND_DESCRIPTIONS[i + 1])
                    dst.set_band_unit(i + 1, BAND_UNITS[i + 1])
            print(f"Prediction completed for tile {cell_id}")
        except Exception as e:
            print("Failed on", cell_id, e)
    return cell_id


def riowarp(
    poly,
    source_rast,
    target_rast,
    t_srs="EPSG:2992",
    nodata=-9999.0,
    xyres=32.808398950125124,  # res in feet
    workers=30,
):
    """Warp raster"""
    proc = subprocess.Popen(
        [
            f"gdalwarp -cutline {poly} -crop_to_cutline -t_srs {t_srs} {source_rast} {target_rast} \
                -of COG -co BIGTIFF=YES -srcnodata {nodata} -tr {xyres} {xyres} -multi -wo {workers}"
        ],
        shell=True,
        text=True,
    )
    output = proc.communicate()
    return output


# %%
if __name__ == "__main__":

    # %%
    YEAR = 2023
    run_as = "prod"

    # conf = ConfigLoader(Path(__file__).parent.parent).load()
    datadir = Path(config.DATADIR)
    images_path = datadir / "processed/tiles"
    grid = gpd.read_file(config.GRID)
    grid.sort_values("CELL_ID", inplace=True)
    if run_as == "dev":
        # datadir = Path(config.DEV_DATADIR)
        to_run = [
            Path(f).stem.split("_")[0]
            for f in image_collection(
                Path(config.DEV_DATADIR) / f"interim/oregon_imagery"
            )
        ]
        # grid = grid.sort_values('CELL_ID').iloc[:10]
        grid = grid[grid.CELL_ID.astype(str).isin(to_run)]
        # images_path = datadir / 'interim/oregon_imagery'

    pred_path = datadir / f"interim/predictions/structure/{YEAR}"
    pred_path.mkdir(exist_ok=True, parents=True)

    sourcer = pred_path.parent / "structure_2023.vrt"
    poly_wa = images_path.parent / "us_states/WA_epsg4326.geojson"
    targetr_wa = (
        images_path.parent / f"predictions/structure/{YEAR}/wa_structure_{YEAR}.tif"
    )
    poly_or = images_path.parent / "us_states/OR_epsg4326.geojson"
    targetr_or = (
        images_path.parent / f"predictions/structure/{YEAR}/or_structure_{YEAR}.tif"
    )

    # %%
    MODEL = (
        Path(config.DATADIR)
        / "models/structure/global-sentinel-HistGradientBoostingRegressor-multioutput.pkl"
    )
    model = pickle.load(open(MODEL, "rb"))

    params = [
        {
            "s2_path": images_path
            / f"sentinel2srh/{row.CELL_ID}_{YEAR}_Sentinel2SRH_leafon-cog.tif",
            "ltr_path": images_path
            / f"landtrendr/{YEAR}/{row.CELL_ID}_{YEAR}_LandTrendr-cog.tif",
            "dem_path": images_path / f"3dep/{row.CELL_ID}_3DEP_10mDEM-cog.tif",
            "dw_path": images_path
            / f"dynamic_world/{YEAR}/{row.CELL_ID}_{YEAR}_DynamicWorld_leafon-cog.tif",
            "outfile": pred_path / f"{row.CELL_ID}_structure_{YEAR}.tif",
            "model": model,
            "bounds": row.geometry.bounds,
            "overwrite": False,
        }
        for row in grid.itertuples()
    ]

    # Generate predictions
    multithreaded_execution(load_predict_save, params, 10)
    # load_predict_save(**params[1])

    # Generate VRT file
    print("Generating VRT file...")
    subprocess.Popen(
        [f"cd {pred_path.parent}; gdalbuildvrt structure_{YEAR}.vrt {YEAR}/*.tif"],
        shell=True,
    )

    # Generate state COGs
    riowarp(
        poly_wa.as_posix(), sourcer.as_posix(), targetr_wa.as_posix(), t_srs="EPSG:2927"
    )
    riowarp(
        poly_or.as_posix(), sourcer.as_posix(), targetr_or.as_posix(), t_srs="EPSG:2992"
    )

    print("Done!")

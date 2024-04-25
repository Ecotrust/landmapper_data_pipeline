import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio import transform, warp
from rasterio.plot import reshape_as_image, reshape_as_raster
from skimage.transform import resize


def load_sentinel(to_load, bounds=None, height=None, width=None, season="leafon", reproject_to=None):
    """Loads and transforms SENTINEL-2 image into a DataFrame.

    Parameters
    ----------
    to_load : str or arr
      path to raster or an in-memory array with SENTINEL-2 data
    bounds : tuple, optional
      (minx, miny, maxx, maxy) bounding box to load

    Returns
    -------
    df : DataFrame
      flattened raster with additional derived fields
    """
    window = None
    if isinstance(to_load, (str, Path)):
        with rasterio.open(to_load) as src:
            if bounds:
                window = src.window(*bounds)
            s2 = src.read(window=window)
    elif isinstance(to_load, np.ndarray):
        s2 = to_load
    else:
        raise TypeError

    if height is not None and width is not None:
        s2 = reshape_as_image(s2)
        s2 = resize(s2, (height, width), order=0, preserve_range=True)
        s2 = reshape_as_raster(s2)

    COLS = [
        "S2_B_LEAFOFF",
        "S2_G_LEAFOFF",
        "S2_R_LEAFOFF",
        "S2_RE1_LEAFOFF",
        "S2_RE2_LEAFOFF",
        "S2_RE3_LEAFOFF",
        "S2_NIR_LEAFOFF",
        "S2_RE4_LEAFOFF",
        "S2_SWIR1_LEAFOFF",
        "S2_SWIR2_LEAFOFF",
        "S2_B_LEAFON",
        "S2_G_LEAFON",
        "S2_R_LEAFON",
        "S2_RE1_LEAFON",
        "S2_RE2_LEAFON",
        "S2_RE3_LEAFON",
        "S2_NIR_LEAFON",
        "S2_RE4_LEAFON",
        "S2_SWIR1_LEAFON",
        "S2_SWIR2_LEAFON",
    ]

    if season == "leafon":
        USE_COLS = [col for col in COLS if "LEAFON" in col]
    if season == "leafoff":
        USE_COLS = [col for col in COLS if "LEAFOFF" in col]
    if season == "both":
        USE_COLS = COLS
    df = pd.DataFrame(s2.reshape((len(USE_COLS), -1)).T, columns=USE_COLS)
    df = df.replace(0, np.nan)  # nodata represented as zeros

    # calculate and add derived metrics in each season
    if season == "leafon":
        seasons = ["LEAFON"]
    if season == "leafoff":
        seasons = ["LEAFOFF"]
    if season == "both":
        seasons = ["LEAFOFF", "LEAFON"]

    for s in seasons:
        R, G, B = f"S2_R_{s}", f"S2_G_{s}", f"S2_B_{s}"
        NIR, SWIR1, SWIR2 = f"S2_NIR_{s}", f"S2_SWIR1_{s}", f"S2_SWIR2_{s}"

        NDVI = f"S2_NDVI_{s}"
        df[NDVI] = (df[NIR] - df[R]) / (df[NIR] + df[R])

        ENDVI = f"S2_ENDVI_{s}"
        df[ENDVI] = (df[NIR] + df[G] - 2 * df[B]) / (df[NIR] + df[G] + 2 * df[B])

        SAVI = f"S2_SAVI_{s}"
        df[SAVI] = 1.5 * (df[NIR] - df[R]) / (df[NIR] + df[R] + 0.5)

        BRIGHTNESS = f"S2_BRIGHTNESS_{s}"
        df[BRIGHTNESS] = (
            0.3029 * df[B]
            + 0.2786 * df[G]
            + 0.4733 * df[R]
            + 0.5599 * df[NIR]
            + 0.508 * df[SWIR1]
            + 0.1872 * df[SWIR2]
        )

        GREENNESS = f"S2_GREENNESS_{s}"
        df[GREENNESS] = (
            -0.2941 * df[B]
            + -0.243 * df[G]
            + -0.5424 * df[R]
            + 0.7276 * df[NIR]
            + 0.0713 * df[SWIR1]
            + -0.1608 * df[SWIR2]
        )

        WETNESS = f"S2_WETNESS_{s}"
        df[WETNESS] = (
            0.1511 * df[B]
            + 0.1973 * df[G]
            + 0.3283 * df[R]
            + 0.3407 * df[NIR]
            + -0.7117 * df[SWIR1]
            + -0.4559 * df[SWIR2]
        )

    # calculate metrics for change in spectral and derived values from leafon to leafoff
    if season == "both":
        for col in [
            "B",
            "G",
            "R",
            "RE1",
            "RE2",
            "RE3",
            "NIR",
            "RE4",
            "SWIR1",
            "SWIR2",
            "NDVI",
            "SAVI",
            "BRIGHTNESS",
            "GREENNESS",
            "WETNESS",
        ]:
            df[f"S2_d{col}"] = df[f"S2_{col}_LEAFON"] - df[f"S2_{col}_LEAFOFF"]

    return df


def load_landtrendr(to_load, bounds=None, height=None, width=None):
    """Loads and transforms a Landtrendr-derived raster into
    into a DataFrame.

    Parameters
    ----------
    to_load : str or arr
      path to raster or an in-memory array with DEM data

    Returns
    -------
    df : DataFrame
      flattened raster with additional derived fields
    """
    window = None
    out_shape = None
    if isinstance(to_load, (str, Path)):
        with rasterio.open(to_load) as src:
            if height is not None and width is not None:
                out_shape = (src.count, src.height, src.width)
            if bounds:
                window = src.window(*bounds)
            lt = src.read(window=window, out_shape=out_shape)
    elif isinstance(to_load, np.ndarray):
        lt = to_load
    else:
        raise TypeError

    if height is not None and width is not None:
        lt = reshape_as_image(lt)
        lt = resize(lt, (height, width), order=0, preserve_range=True)
        lt = reshape_as_raster(lt)

    COLS = [
        "LT_YSD_SWIR1",
        "LT_MAG_SWIR1",
        "LT_DUR_SWIR1",
        "LT_RATE_SWIR1",
        "LT_YSD_NBR",
        "LT_MAG_NBR",
        "LT_DUR_NBR",
        "LT_RATE_NBR",
    ]

    df = pd.DataFrame(lt.reshape([8, -1]).T, columns=COLS, dtype="Int64").replace(
        -32768, np.nan
    )  # nodata represented as -32768

    return df


def load_dem(to_load, profile=None, bounds=None):
    """Loads and transforms a Digital Elevation Model (DEM) image
    into a DataFrame.

    Parameters
    ----------
    to_load : str or arr
      path to raster or an in-memory array with DEM data
    meta : dict, optional
      dictionary of raster attributes, must include width,
      height, transform, and crs

    Returns
    -------
    df : DataFrame
      flattened raster with additional derived fields
    """
    window = None
    affine = None
    crs = None
    if isinstance(to_load, (str, Path)):
        with rasterio.open(to_load) as src:
            if bounds:
                window = src.window(*bounds)
                affine = src.window_transform(window)
            else:
                affine = src.transform
            crs = src.crs
            dem = src.read(1, window=window)
            # meta = src.meta
    elif isinstance(to_load, np.ndarray):
        assert profile is not None, "profile must be provided when passing an array"
        dem = to_load
        crs = profile['crs']
        affine = profile['transform']
    else:
        raise TypeError

    # if height is not None and width is not None and meta is not None:
    #     dem = resize(dem, (height, width), order=0, preserve_range=True)
    #     meta["height"] = height
    #     meta["width"] = width

    df = pd.DataFrame(columns=["ELEVATION", "LAT", "LON"])

    df["ELEVATION"] = dem.ravel()
    df["ELEVATION"] = df["ELEVATION"].astype(np.int64)

    # fetch lat and lon for each pixel in a raster
    rows, cols = np.indices(dem.shape)
    xs, ys = transform.xy(affine, cols.ravel(), rows.ravel())
    if crs != 4326:
        xs, ys = warp.transform(crs, {"init": "EPSG:4326"}, xs, ys)

    df["LAT"] = ys
    df["LON"] = xs

    # nodata represented as -32768
    df.loc[df.ELEVATION == -9999] = np.nan

    return df


def load_nlcd(to_load, year=None, height=None, width=None):
    """Loads and transforms a NLCD image into a DataFrame.

    Parameters
    ----------
    to_load : str or arr
      path to raster or an in-memory array with NLCD data

    Returns
    -------
    df : DataFrame
      flattened rasters
    """
    if isinstance(to_load, str):
        with rasterio.open(to_load) as src:
            nlcd = src.read(1)
    elif isinstance(to_load, np.ndarray):
        nlcd = to_load

    else:
        raise TypeError

    if height is not None and width is not None:
        nlcd = resize(nlcd, (height, width), order=0, preserve_range=True)

    df = pd.DataFrame(dtype="Int64")

    if year is not None:
        df[f"NLCD_{year}"] = nlcd.ravel()
        df.loc[df[f"NLCD_{year}"] == 0] = np.nan
    else:
        df["NLCD"] = nlcd.ravel()
        df.loc[df.NLCD == 0] = np.nan

    return df


def load_structure(to_load, height=None, width=None):
    """Loads and transforms a forest structure prediction raster image into a DataFrame.

    Parameters
    ----------
    to_load : str or arr
      path to raster or an in-memory array with predicted forest structure data

    Returns
    -------
    df : DataFrame
      flattened raster
    """
    if isinstance(to_load, str):
        with rasterio.open(to_load) as src:
            img = src.read()
    elif isinstance(to_load, np.ndarray):
        img = to_load
    else:
        raise TypeError

    if height is not None and width is not None:
        img = reshape_as_image(img)
        img = resize(img, (height, width), order=0, preserve_range=True)
        img = reshape_as_raster(img)

    COLS = ["TOTAL_COVER", "TOPHT", "QMD", "SDI", "TCUFT", "ABOVEGROUND_BIOMASS"]

    df = pd.DataFrame(img.reshape([6, -1]).T, columns=COLS, dtype="Int64").replace(
        0, np.nan
    )  # nodata represented as 0

    return df

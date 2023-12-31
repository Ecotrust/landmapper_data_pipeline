import os
import numpy as np
import pandas as pd
import rasterio
from rasterio import transform, warp
from rasterio.plot import reshape_as_image, reshape_as_raster
from skimage.transform import resize


def load_sentinel(to_load, height=None, width=None, season="leafon"):
    """Loads and transforms SENTINEL-2 image into a DataFrame.

    Parameters
    ----------
    to_load : str or arr
      path to raster or an in-memory array with SENTINEL-2 data

    Returns
    -------
    df : DataFrame
      flattened raster with additional derived fields
    """
    if isinstance(to_load, str):
        with rasterio.open(to_load) as src:
            s2 = src.read()
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
        "S2_RE4_LEAFOFF",
        "S2_NIR_LEAFOFF",
        "S2_SWIR1_LEAFOFF",
        "S2_SWIR2_LEAFOFF",
        "S2_B_LEAFON",
        "S2_G_LEAFON",
        "S2_R_LEAFON",
        "S2_RE1_LEAFON",
        "S2_RE2_LEAFON",
        "S2_RE3_LEAFON",
        "S2_RE4_LEAFON",
        "S2_NIR_LEAFON",
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
            "RE4",
            "NIR",
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


def load_landtrendr(to_load, height=None, width=None):
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
    if isinstance(to_load, str):
        with rasterio.open(to_load) as src:
            lt = src.read()
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


def load_dem(to_load, meta=None, height=None, width=None):
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
    if isinstance(to_load, str):
        with rasterio.open(to_load) as src:
            dem = src.read(1)
            meta = src.meta
    elif isinstance(to_load, np.ndarray) and meta is not None:
        dem = to_load

    else:
        raise TypeError

    if height is not None and width is not None and meta is not None:
        dem = resize(dem, (height, width), order=0, preserve_range=True)
        meta["height"] = height
        meta["width"] = width

    df = pd.DataFrame(columns=["ELEVATION", "LAT", "LON"])

    df["ELEVATION"] = dem.ravel()
    df["ELEVATION"] = df["ELEVATION"].astype("Int64")

    # fetch lat and lon for each pixel in a raster
    rows, cols = np.indices((meta["height"], meta["width"]))
    xs, ys = transform.xy(meta["transform"], cols.ravel(), rows.ravel())
    lons, lats = warp.transform(meta["crs"], {"init": "EPSG:4326"}, xs, ys)
    df["LAT"] = lats
    df["LON"] = lons

    # nodata represented as -32768
    df.loc[df.ELEVATION == -32768] = np.nan

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


def load_features(path_to_rasters, cell_id, year, season="leafon"):
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
    s2_path = os.path.join(path_to_rasters, "sentinel", f"{cell_id}_sentinel{year}.tif")
    lt_path = os.path.join(
        path_to_rasters, "landtrendr", f"{cell_id}_landtrendr{year}.tif"
    )
    dem_path = os.path.join(path_to_rasters, "dem", f"{cell_id}_dem.tif")

    s2 = load_sentinel(s2_path, season=season)
    lt = load_landtrendr(lt_path)
    dem = load_dem(dem_path)

    df = pd.concat([s2, lt, dem], axis=1)

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

    return df[COL_ORDER]

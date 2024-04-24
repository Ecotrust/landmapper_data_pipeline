# %%
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio import MemoryFile
from rasterio.merge import merge
import rasterio.mask
from rasterio.warp import Resampling
from rasterio.crs import CRS

from gdstools import (
    image_collection,
    html_to_rgb,
    ConfigLoader,
    save_cog,
)

# %%
conf = ConfigLoader(Path(__file__).parent).load()
model_name = "train_f-climate_prod172023051012"
traindts = conf.labels
preds_path = Path(conf.DATADIR) / "predictions" / model_name
mosaic_path = preds_path.parent / "mosaic"
mosaic_path.mkdir(exist_ok=True)
mosaic_name = mosaic_path / f"{model_name.replace('train', 'mosaic')}.tif"
grid = Path(conf.GRID)
# RES = 8.983152656583115e-05*10
RES = None

collection = image_collection(preds_path)
colormap = traindts["colormap"]
colormap.update({14: "#c9c9c9"})
labels = traindts["labels"]
labels.update({14: "Non-forest"})
colordict = {k: html_to_rgb(c) + (255,) for k, c in colormap.items()}
categories = ",".join([f"{k}: {v}" for k, v in labels.items()])

# %%
# for st in ["OR", "WA"]:
mosaic_name = mosaic_path / f"{model_name.replace('train', 'orwa_forest_types')}.tif"
stgrid = gpd.read_file(grid)
st_collection = [
    p for p in collection if int(Path(p).name.split("_")[0]) in stgrid.CELL_ID.tolist()
]

# predictions = []
# for pred in st_collection:
#     src = rasterio.open(pred)
#     predictions.append(src)

predictions = []
for row in stgrid.itertuples():
    row.CELL_ID
    # path = [p for p in st_collection if int(Path(p).name.split("_")[0]) == cell_id][0]
    path = preds_path / f'{row.CELL_ID}_forest_type_predictions.tif'
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        window = rasterio.windows.from_bounds(*row.geometry.bounds, transform=src.transform)
        profile.update(
            {
                "driver": "GTiff",
                "crs": CRS.from_epsg(4326),
                "height": window.height,
                "width": window.width,
                "transform": src.window_transform(window),
                "resampling": Resampling.nearest,
                "count": 1,
            }
        )

        with MemoryFile() as memfile:
            dst = memfile.open(**profile)
            dst.write(src.read(window=window))
            predictions.append(dst)

    
# %%
mosaic, transform = merge(predictions, res=RES)
# show(mosaic)

bbox = stgrid.total_bounds
h, w = mosaic.shape[1:3]
profile = src.profile.copy()
profile.update(
    {
        "driver": "GTiff",
        "crs": CRS.from_epsg(4326),
        "height": h,
        "width": w,
        "transform": transform,
        "resampling": Resampling.nearest,
        "count": 1,
    }
)

# %%
save_cog(
    mosaic,
    profile,
    mosaic_name,
    colordict=colordict,
    categories=categories,
    overwrite=False,
)

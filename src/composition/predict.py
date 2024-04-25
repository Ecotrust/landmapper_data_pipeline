# %%
import os
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision import transforms
from torch.utils.data import DataLoader

# from torchmetrics import JaccardIndex, ConfusionMatrix

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio import Affine, MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from tqdm import tqdm

from gdstools import (
    Denormalize,
    multithreaded_execution,
    ConfigLoader,
    image_collection,
    html_to_rgb,
    save_cog,
)
from src.composition.dataloader import fSegNetDataset
from src.composition.models import UNet

# %%
conf = ConfigLoader(Path(__file__).parent).load()

YEAR = 2023
GPUID = 0
DEVICE_LIST = [0, 1, 2]
PARALLEL = True
WORKERS = 4
SIZE = None  # size of testing tiles
BATCH_SIZE = 5
ND_CLASS = 0
MODEL_NAME = "train_f-climate_prod172023051012"

root = Path(conf.DATADIR) / "processed/tiles"
outdir = Path(conf.DATADIR) / f"predictions/forest_types/{MODEL_NAME}_{YEAR}"
outdir.mkdir(parents=True, exist_ok=True)
modelpath = Path(conf.DATADIR) / f"models/composition/{MODEL_NAME}.pth"
or_poly = root / "processed/us_states/OR_epsg4326.geojson"
wa_poly = root / "processed/us_states/WA_epsg4326.geojson"
source_rast = outdir.parent / f"clipped_predictions_{YEAR}.vrt"
target_rast = outdir.parent / f"clipped_predictions_{YEAR}_2992.tif"


def save_image(image, transform, path, dtype=rasterio.uint8, overwrite=False):
    cog_profile = cog_profiles.get("deflate")
    if not os.path.exists(path) or overwrite:
        crs = CRS.from_epsg(4326)

        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                crs=crs,
                transform=transform,
                height=image.shape[0],
                width=image.shape[1],
                dtype=dtype,
                count=1,
            ) as dst:
                dst.write(image, 1)
                cog_translate(dst, path, cog_profile, in_memory=True, quiet=True)
    else:
        print(f"File {path} already exists")


def match_template(source_raster, template_raster, filename, out_dir="."):
    # cell_id = os.path.basename(template_raster).split("_")[-1]
    # out_file = f"{cell_id}_2017_forest_habitats-cog.tif"
    # out_file = f"{filename}_{cell_id}"
    out_path = os.path.join(out_dir, filename)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "rio",
            "warp",
            "-o",
            out_path,
            "--like",
            template_raster,
            source_raster,
            # "data/external/ownership/forest_own1.tif",
            # "./gnn_orwa_habtypes.tif",
        ],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    return proc


def get_predictions():
    device = torch.device(f"cuda:{GPUID}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Add indexes of bands to use for each input layer during training
    use_bands = {
        "sentinel2sr": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "climatena": [13, 16, 23],
        "3dep": [1, 2, 3],
    }
    input_lyrs = list(use_bands.keys())
    label_conf = conf["labels"]
    labels = label_conf["labels"]
    outchannels = len(labels.keys())
    df = pd.read_csv(root / "metadata.csv")
    df = df[df.collection.isin(input_lyrs)]
    df = df[df.year.isin([YEAR, 9999])]
    # The stats dict contains mean and std for all image bands. The full image stack
    # is composed of 46 bands, but we only use a subset of them for training.
    # We need to select the mean and std for the bands we use.
    indexes = (
        (np.array(use_bands["sentinel2sr"]) - 1).tolist()
        + (np.array(use_bands["climatena"]) - 1 + 10).tolist()
        + (np.array(use_bands["3dep"]) - 1 + 35).tolist()
        + [43]
    )
    inchannels = len(indexes)
    stats = conf.get("stats")
    _mean = [stats["mean"][i] for i in indexes]
    _std = [stats["std"][i] for i in indexes]
    normalize = transforms.Normalize(mean=_mean, std=_std)
    revert = Denormalize(mean=_mean, std=_std)

    dts = fSegNetDataset(
        task="inference",
        root=root,
        dataframe=df,
        size=SIZE,
        use_bands=use_bands,
        input_layers=input_lyrs,
        transform=transforms.Compose([normalize]),
    )
    dataloader = DataLoader(
        dts, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=False
    )

    model = UNet(in_channels=inchannels, out_channels=outchannels)
    model = model.to(device)

    # If the model was trained with nn.DataParallel, it needs to be
    # loaded with nn.DataParallel as well, otherwise it will complain about
    # missing parameters.
    if PARALLEL:
        model = nn.DataParallel(model, device_ids=DEVICE_LIST)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    with torch.no_grad():
        model.eval()
        test_loop = tqdm(dataloader)
        for batch in test_loop:
            images = batch["image"].to(device)
            filenames = [
                f"{outdir}/{x}_forest_type_predictions.tif" for x in batch["cellid"]
            ]
            transforms = [Affine(*x.numpy()) for x in batch["transform"]]

            pred_prob = model(images)
            pred_prob = F.pad(input=pred_prob, pad=(6, 6, 6, 6), mode="constant")

            sem_proba = F.softmax(pred_prob, dim=1)
            predictions = sem_proba.argmax(dim=1) + 1

            params = [
                {
                    "image": i.cpu().numpy(),
                    "transform": t,
                    "path": p,
                    "dtype": rasterio.uint8,
                    "overwrite": True,
                }
                for i, t, p in zip(predictions, transforms, filenames)
            ]

            multithreaded_execution(save_image, params, threads=len(params))

    return


def clip_prediction_tiles():
    # Clip predictions to the extent of 7.5 quarter quad tiles
    unclipped = image_collection(outdir)
    clipped_path = outdir.parent / f"clipped_predictions_{YEAR}"

    params = [
        {
            "source_raster": p,
            "template_raster": outdir.parent / MODEL_NAME / f"{Path(p).stem}.tif",
            "filename": f"{Path(p).stem}.tif",
            "out_dir": clipped_path,
        }
        for p in unclipped
    ]

    multithreaded_execution(match_template, params)

    clipped = image_collection(clipped_path)
    if len(unclipped) == len(clipped):
        print(
            f"Unclipped predictions: {len(unclipped)}",
            f"Clipped predictions: {len(clipped)}",
        )
        # Remove uncliped tiled predictions
        print("Removing unclipped predictions...")
        outdir.unlink()

        # Generate VRT file for clipped predictions
        print("Generating VRT file...")
        subprocess.Popen(
            [
                f"cd {outdir.parent}; gdalbuildvrt clipped_predictions_{YEAR}.vrt clipped_predictions_{YEAR}/*.tif"
            ],
            shell=True,
        )

        print("Clipping predictions completed!")
    else:
        print("Number of clipped predictions do not match original predictions.")

    return


def create_state_mosaic(overwrite=False):
    states = image_collection(
        Path(conf.DATADIR) / "processed/us_states", file_pattern="*.geojson"
    )
    vrt = (
        Path(conf.DATADIR)
        / f"interim/predictions/forest_types/clipped_predictions_{YEAR}.vrt"
    )
    traindts = conf.labels
    colormap = traindts["colormap"]
    colormap.update({14: "#c9c9c9"})
    labels = traindts["labels"]
    labels.update({14: "Non-forest"})
    colordict = {k: html_to_rgb(c) + (255,) for k, c in colormap.items()}
    categories = ",".join([f"{k}: {v}" for k, v in labels.items()])

    for state in states:
        poly = gpd.read_file(state)
        st_pre = Path(state).stem.split("_")[0].lower()
        print(f"Processing mosaic for {st_pre.upper()} state...")
        outpath = (
            Path(conf.DATADIR)
            / f"processed/predictions/forest_types/{YEAR}/{st_pre}_forest_types_{YEAR}.tif"
        )
        if outpath.exists() and overwrite:
            continue

        with rasterio.open(vrt) as src:
            window = src.window(*poly.geometry[0].bounds)
            predictions = src.read(window=window, masked=True)
            w_transform = src.window_transform(window)
            w_bounds = src.window_bounds(window)
            h, w = predictions.shape[1:3]
            profile = src.profile.copy()
            mask = features.rasterize(
                [(poly.geometry[0], 1)],
                out_shape=predictions.shape[1:],
                transform=w_transform,
                fill=0,
                all_touched=True,
                dtype="uint8",
            )
            predictions.mask = mask == 0
            del mask

            # Reproject to EPSG:2992
            dst_crs = CRS.from_epsg(2992)
            if st_pre == "wa":
                dst_crs = CRS.from_epsg(2927)
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, w, h, *w_bounds, resolution=10
            )

            geom = poly.to_crs(dst_crs).geometry[0]
            mask = features.rasterize(
                [(geom, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype="uint8",
            )
            proj_preds = np.ma.zeros((1, height, width), dtype=rasterio.uint8)
            proj_preds.mask = mask == 0
            del mask

            reproject(
                source=predictions,
                destination=proj_preds,
                src_transform=w_transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )

            profile.update(
                {
                    "driver": "GTiff",
                    "crs": dst_crs,
                    "height": height,
                    "width": width,
                    "blockxsize": 512,
                    "blockysize": 512,
                    "transform": transform,
                    "resampling": Resampling.nearest,
                    "count": 1,
                    "nodata": 0,
                }
            )

            save_cog(
                proj_preds,
                profile,
                outpath,
                # colordict=colordict,
                categories=categories,
                overwrite=True,
            )

        print(f"Prediction mosaic for {st_pre.upper()} state completed!")

    return


def riowarp(
    poly,
    source_rast,
    target_rast,
    t_srs="EPSG:2992",
    nodata=0,
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


if __name__ == "__main__":
    get_predictions()
    clip_prediction_tiles()
    create_state_mosaic()
    print("All done!")

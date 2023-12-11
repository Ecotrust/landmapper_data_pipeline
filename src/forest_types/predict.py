# %%
import os
from rasterio.crs import CRS
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, ConfusionMatrix

import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine
from rasterio import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from tqdm import tqdm

from src_pipelines.utils import (
    Denormalize,
    multithreaded_execution,
    ConfigLoader,
)
from src_pipelines.forest_types.dataloader import fSegNetDataset
from src_pipelines.forest_types.models import UNet

# %%
conf = ConfigLoader(Path(__file__).parent).load()

MODEL_NAME = "train_f-climate_prod172023051012"
GPUID = 0
DEVICE_LIST = [0, 1, 2]
PARALLEL = True
WORKERS = 4
SIZE = 840  # size of testing tiles
BATCH_SIZE = 5
ND_CLASS = 0

device = torch.device(f"cuda:{GPUID}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)
root = Path(conf.DATADIR) / "tiles"
outdir = Path(conf.DATADIR) / f"predictions/{MODEL_NAME}"
modelpath = f"src_pipelines/forest_types/trained_models/{MODEL_NAME}.pth"

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

# %%
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
    dts, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True
)

# %%
model = UNet(in_channels=inchannels, out_channels=outchannels)
model = model.to(device)

# It seems that if the model was trained with nn.DataParallel, it needs to be
# loaded with nn.DataParallel as well, otherwise it will complain about 
# missing parameters.
if PARALLEL:
    model = nn.DataParallel(model, device_ids=DEVICE_LIST)
model.load_state_dict(
    torch.load(modelpath)
)
model.eval()

jaccard = JaccardIndex(
    task="multiclass", num_classes=outchannels + 1, ignore_index=ND_CLASS
).to(device)
confusion = ConfusionMatrix(task="multiclass", num_classes=outchannels + 1).to(device)


# %%
def save_image(image, transform, path, dtype=rasterio.uint8):
    cog_profile = cog_profiles.get("deflate")
    if not os.path.exists(path):
        crs = CRS.from_epsg(4326)

        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                crs=crs,
                transform=transform,
                height=image.shape[0],
                width=image.shape[1],
                dtype=rasterio.float32,
                count=1,
            ) as dst:
                dst.write(image, 1)
                cog_translate(dst, path, cog_profile, in_memory=True, quiet=True)
    else:
        print(f"File {path} already exists")

# %%
outdir.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    model.eval()
    test_loop = tqdm(dataloader)
    for batch in test_loop:
        images = batch["image"].to(device)
        filenames = [f"{outdir}/{x}_forest_type_predictions.tif" for x in batch["cellid"]]
        transforms = [Affine(*x.numpy()) for x in batch["transform"]]

        pred_prob = model(images)

        sem_proba = F.softmax(pred_prob, dim=1)
        predictions = sem_proba.argmax(dim=1) + 1

        params = [
            {
                "image": i.cpu().numpy(),
                "transform": t,
                "path": p,
                "dtype": rasterio.uint8,
            }
            for i, t, p in zip(predictions, transforms, filenames)
        ]

        multithreaded_execution(save_image, params, threads=len(params))

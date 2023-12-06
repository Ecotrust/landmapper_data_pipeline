#!/usr/bin/python3

import os

import numpy as np
import rasterio as rio
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


def html_to_rgb(html_color):
    # Remove the '#' character from the beginning of the HTML color string
    html_color = html_color.lstrip('#')

    # Convert the HTML color string to a tuple of integers
    rgb_tuple = tuple(int(html_color[i:i+2], 16) for i in (0, 2, 4))

    return rgb_tuple


def image_collection(path, file_pattern="*.tif"):
    import glob

    return glob.glob(f"{path}/**/{file_pattern}", recursive=True)


def save_cog(
    data: np.ndarray,
    profile: dict,
    path,
    colordict=None,
    categories=None,
    overwrite=False,
):
    cog_profile = cog_profiles.get("deflate")
    if not os.path.exists(path) or overwrite:
        with rio.MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(data)

                if colordict is not None:
                    dst.write_colormap(1, colordict)
                if categories is not None:
                    dst.update_tags(CATEGORY_NAMES=categories)

                cog_translate(dst, path, cog_profile, in_memory=True, quiet=False)
    else:
        print(f"File {path} already exists")

    return


class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, y):
        x = y.new(*y.size())
        for i in range(x.shape[0]):
            x[i, :, :] = y[i, :, :] * self.std[i] + self.mean[i]
        return x

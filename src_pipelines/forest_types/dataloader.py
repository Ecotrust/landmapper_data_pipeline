# %%
from pathlib import Path
from typing import Callable, Dict, Optional, List

import torch
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class fSegNetDataset(Dataset):
    def __init__(
        self,
        task: str,
        root: str,
        dataframe: pd.DataFrame,
        input_layers: List[str],
        reclass_dict: Dict[int, List[int]] = None,
        size: int = 256,
        samples_per_epoch: int = 5,
        random_state: int = 42,
        transform: Optional[Callable] = None,
        use_bands: Dict[str, List[str]] = None,
    ) -> None:
        """Initialize a Forest Type Prediction dataset.

        Args:
            task (str): The task to perform. One of ['train', 'inference'].
            dataframe (pd.DataFrame): A dataframe containing input and target medata.
            input_layers (List[str], optional): A list of input layers to use. Defaults to ['sentinel2'].
            reclass_dict (Dict[int, List[int]], optional): A dictionary {class: newclass} of reclassifications. Defaults to None.
            stats (Dict[str, Dict[str, float]], optional): A dictionary of statistics for each input layer. Defaults to None.
            size (int, optional): The size of the image. Defaults to 256.
            samples_per_epoch (int, optional): The number of samples per epoch. Defaults to 5.
            random_state (int, optional): The random state. Defaults to 42.
            transform (Optional[Callable], optional): A transform to apply to the data. Defaults to None.
        """
        super().__init__()
        self.task = task
        self.root = Path(root)
        self.size = size
        self.samples_per_epoch = samples_per_epoch
        self.df = dataframe
        self.cellids = dataframe.cellid.unique().tolist()
        self.input_layers = input_layers
        self.reclass = None
        self.random_state = random_state
        self.transform = transform
        self.use_bands = use_bands

        if reclass_dict:
            self.reclass = reclass_dict

    def __getitem__(self, index):
        cellid = self.cellids[index]
        images = self.df.query("cellid == @cellid & target == 0").to_dict("records")

        info = images[0]
        xmax = info["width"] - self.size
        ymax = info["height"] - self.size
        xoff = xmax // 2
        yoff = ymax // 2
        window = Window(xoff, yoff, self.size, self.size)

        centroid_x = int(1e5 * info["centroid_x"])
        centroid_y = int(1e5 * info["centroid_y"])

        sample = {}

        if self.task == "train":
            try:
                target_dict = (
                    self.df.query(
                        "cellid == @cellid & "
                        "target == 1 & "
                        'collection == "forest_habitat"'
                    )
                    .to_dict("records")
                    .pop()
                )
                dw_dict = (
                    self.df.query(
                        "cellid == @cellid & "
                        "target == 1 & "
                        'collection == "dynamic_world"'
                    )
                    .to_dict("records")
                    .pop()
                )
            except IndexError:
                raise IndexError(f"Cellid {cellid} does not have a target image.")

            fcls = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

            with rasterio.open(target_dict["filepath"]) as src:
                src_target = src.read(1, window=window)

                target = np.zeros((window.width, window.height), dtype=np.uint8)
                
                if self.reclass:
                    reclass = {}
                    # First get boolean arrays with all the indexes to be reclassed
                    for k, v in self.reclass.items():
                        reclass[k] = np.isin(src_target, v)
                    for k, v in reclass.items():
                        target[v] = k
                else:
                    target = src_target

            sample["nodata"] = torch.BoolTensor(target == 0).unsqueeze(0)
            sample["mask"] = torch.LongTensor(target).unsqueeze(0)

        input_data = []
        profiles = []
        for layer in self.input_layers:
            assert (
                layer in self.input_layers
            ), f"Layer {layer} not found in input layers"

            try:
                image_path = [
                    d["filepath"] for d in images if d["collection"] == layer
                ][0]
            except IndexError:
                raise IndexError(f"Layer {layer} not found in input layers")

            with rasterio.open(image_path) as src:
                profiles.append(src.profile)
                if self.use_bands:
                    try:
                        bands = self.use_bands[layer]
                        data = src.read(bands, window=window)
                    except KeyError:
                        raise KeyError(f"Collection {layer} not found.")
                else:
                    data = src.read(window=window)

                input_data.append(data)

        # Compute new transform
        new_origin = rasterio.transform.xy(
            profiles[0]["transform"], yoff, xoff, offset="ul"
        )
        pxx = profiles[0]["transform"][0]
        # pxy = profiles[0]['transform'][1]
        new_transform = rasterio.transform.from_origin(*new_origin, pxx, pxx)

        # Create centroid latlon layer to add as a channel to input images
        latlon = np.zeros((window.width, window.height), dtype=np.longlong)
        latlon.fill(centroid_x)
        input_data.append(np.expand_dims(latlon, 0))

        image = torch.FloatTensor(np.vstack(input_data))
        if self.transform:
            image = self.transform(image)

        sample["image"] = image
        sample["cellid"] = cellid
        sample["transform"] = torch.FloatTensor([*new_transform])

        return sample

    def __len__(self) -> int:
        return len(self.cellids)


# %%
if __name__ == "__main__":
    # TODO: Add tests
    # validate_dataloader()
    from src_pipelines.utils import ConfigLoader

    params = ConfigLoader(Path(__file__).parent)
    conf = params.load()
    root = Path(conf.PROJDIR) / "data/dev"

    task = "train"
    device = "cuda:1"
    torch.cuda.set_device(device)

    df = pd.read_csv(root / "metadata.csv")
    # df = df.query('dataset == "train"')
    use_bands = {
        "sentinel2har": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
        "climatena": [13, 16, 23],
        "3dep": [1, 2, 3],
    }
    ftpdataset = fSegNetDataset(
        task=task,
        root=root,
        dataframe=df,
        use_bands=use_bands,
        size=336,
        input_layers=["sentinel2har", "climatena", "3dep"],
    )
    dataloader = DataLoader(ftpdataset, batch_size=5, shuffle=False, num_workers=10)

    for batch in dataloader:
        print(batch["image"].to(device).shape)
        if task == "train":
            print(batch["mask"].to(device).shape)
            print(batch["nodata"].to(device).shape)
        break

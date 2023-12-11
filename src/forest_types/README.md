# Predicting Oregon and Washington State Forest Community Types

These collection of scripts serves the purpose of utilizing a trained [forestsegnet](https://githublinktorepo.com) model to generate a new forest type mosaic for Oregon and Washington.

The model predicts the following community types:

| Class | Description |
| --- | --- |
| 1 | Shrub |
| 2 | Riparian |
| 3 | Lodgepole pine |
| 4 | Ponderosa Pine |
| 5 | Mixed Conifer |
| 6 | Western Juniper |
| 7 | Mixed Oak - Conifer |
| 8 | Quaking Aspen |
| 9 | Mixed Hardwood - Conifer |
| 10 | Coastal Spruce Cedar or Redwood |
| 11 | Douglas-fir - Western Hemlock |
| 12 | Silver fir - Mountain Hemlock |
| 13 | Spruce - Subalpine Fir |


## Running the scripts

### 1. Preliminaries

Run the following commands to setup the environment variables in a `.env` file in the `forest_types` directory:

```
echo DATADIR=/path/to/data > forest_types/.env
echo GRID=/path/to/usgs_grid >> forest_types/.env
```

`DATADIR` is the path to the directory containing the input data and the destination for predictions. The following directories and datasets are expected:

    DATADIR
    |── tiles
    |   ├── climatena       <-- 3-band imagery with data from PRISM 1971-2000 gridded monthly climate data
    |   ├── dem             <-- 3-band raster with elevation data from the USGS 3D Elevation Program
    |   ├── sentinelsr      <-- Sentinel-2 surface reflectance imagery with 12-bands
    |   └── dynamic_world   <-- 1-band images with forest/non-forest mask from Dynamic World
    └── predictions         <-- output directory for model predictions

 `GRID` is the path to the USGS quarter quads grid geojson file.

### 2. Generate forest type predictions for each quarter quadrangle tile

Run the script `predict.py`

```
python predict.py
```

### 3. Generate the forest type mosaic

Run the script `create_ft_mosaic.py`

```
python create_ft_mosaic.py
```

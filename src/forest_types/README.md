# Predicting Oregon and Washington State Forest Community Types

These collection of scripts serves the purpose of utilizing a pre-trained model to generate a forest community types raster map for Oregon and Washington.

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
    |   ├── climatena       <-- PRISM 1971-2000 gridded monthly climate data
    |   ├── dem             <-- Elevation data from the USGS 3D Elevation Program
    |   ├── sentinel2sr     <-- Sentinel-2 surface reflectance imagery with 10-bands
    |   └── dynamic_world   <-- Forest/non-forest mask from Dynamic World
    └── predictions         <-- Output directory for model predictions

 `GRID` is the path to the USGS quarter quads grid geojson file.

### 2. Download new input data

`download_dem.py` - Fetchs DEM from the USGS 3D Elevation Program (3DEP) and calculates the following layers: Slope, Aspect, Flow Accumulation, Topographic Position Index (TPI300 and TPI2000), Slope Position Class (SPC300), and Land Form Class. All layers are saved in a single GeoTIFF file with 8 bands.

```bash
python download_dem.py
```

`download_dem.py` - Fetches Sentinel-2 surface reflectance imagery from Google Earth Engine. The bands fetched are: B2, B3, B4, B5, B6, B7, B8, B8A, B11, and B12.

```bash
python download_s2.py --year 2023
```

`download_s2.py` - Fetches a forest/non-forest mask from Dynamic World land cover classification available in Google Earth Engine. 

```bash
python download_dw.py --year 2023
```

### 3. Generate forest type predictions for each quarter quadrangle tile

Run the script `predict.py`

```bash
python predict.py
```

### 4. Generate the forest type mosaic

Run the script `create_ft_mosaic.py`

```bash
python create_ft_mosaic.py
```

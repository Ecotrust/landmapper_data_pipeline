Land Mapper Data Pipeline
========================

The purpose of the Land Mapper Data Pipeline is to streamline the workflow for fetching, processing, and updating geospatial data layers used in [Land Mapper's](https://github.com/Ecotrust/landmapper) mapping platform. It consists of Jupyter notebooks and python scripts for fetching and processing raster and vector data from various public repositories, as well as the code to refresh the map layers with new data. The pipeline integrates code to update the following layers available in Land Mapper: Forest Community Type, Tree Size Class, Canopy Cover, Forest Stocking, Soil Type, and Taxlots.

The repository is organized as follows 

------------

    ├── LICENSE
    ├── README.md          <- The top-level README. You are here.
    |
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks for data pipelines.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis in various formats.
    │
    ├── environment.yml    <- Python dependencies for reproducing the environment.
    │
    ├── setup.py           <- Makes project pip installable so src can be imported.
    |
    └── src                <- Source code for data pipelines.
        |
        └── forest_types   <- Forest Community Types data generation pipeline.
--------

Installation
------------

```bash
git clone https://github.com/Ecotrust/landmapper_data_pipeline.git

cd landmapper_data_pipeline
mamba env create -f environment.yml
mamba activate landmapper-dpl
```
--------

Usage
--------
## Refreshing the Forest Community Types layer

To generate the Forest Community Types map we use a pre-trained UNet model. The model was trained on forest composition data produced by the [LEMMA research group](https://lemma.forestry.oregonstate.edu/data) at Oregon State University for Oregon and Washington State. The input data for the model consists of a stack of Sentinel-2 surface reflectance imagery, an 8-band raster with a DEM and 7 topographic metrics, and an 8-band raster with climate variables from PRISM. 

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
    |   ├── 3dep            <-- Elevation data from the USGS 3D Elevation Program
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
python download_s2.py --year <year>
```

`download_s2.py` - Fetches a forest/non-forest mask from Dynamic World land cover classification available in Google Earth Engine. 

```bash
python download_dw.py --year <year>
```

Note: We will add instruction to download and process PRISM climate data soon.

### 3. Generate forest community type predictions

Run the script `predict.py`

```bash
python predict.py
```

### 4. Generate forest community type mosaic

Run the script `create_ft_mosaic.py`

```bash
python create_ft_mosaic.py
```
----


## Refreshing forest structure map layers

Three forest structure map layers, Tree Size Class, Canopy Cover, and Forest Stocking are generated using a pre-trained Gradient Boosting regression model. The model was trained on forest attributes data produced by the [LEMMA research group](https://lemma.forestry.oregonstate.edu/data) at Oregon State University. The input data for the model consists of a stack of Sentinel-2 surface reflectance imagery, an 8-band raster with a DEM and 7 topographic metrics, and an 8-band raster with climate variables from PRISM.

### 1. Download new input data

Run the following Jupyter notebooks from the notebooks directory:

- Download_Dynamic_World-Oregon.ipynb
- Download_Dynamic_World-Washington.ipynb
- Download_S2_Ltr_DEM-Oregon.ipynb
- Download_S2_Ltr_DEM-Washington.ipynb

### 2. Generate forest structure predictions

Run the following Jupyter notebooks:

- Predict_Forest_Structure-Oregon.ipynb
- Predict_Forest_Structure-Washington.ipynb

## Refreshing the Soil Type layer

Instructions coming soon.

--------

## Refreshing the Taxlots layer

Instructions coming soon.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

landmapper_data_pipeline
==============================

Data fetching and data generation pipelines for LandMapper

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yaml   <- Python dependencies for reproducing the environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |
    └── src                <- Source code for data pipelines.
        |
        └── forest_types   <- Forest Types data generation pipeline.

--------

Add dotenv file with required environment variables.

    echo DATADIR=/path/to/datadir/ > .env

Setup dev environment with mamba:

    mamba env create -f environment.yml
    conda activate landmapper-dpl

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

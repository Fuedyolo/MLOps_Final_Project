Pytorch Geometric for classification of CiteSeer papers
==============================

This repository contains the project work carried out by group 27 (Lucie Fontaine, Mathias Sofus Hovmark and Frederik Ommundsen) in the MLOps course taught at DTU. 

1. **Overall goal**: The overall goal is to use the Citeseer dataset and predict which scientific class each node belongs to using a GNN.

2. **What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)**: We are going to use the Pytorch Geometrics framework as we are building a GNN.

3. **How to you intend to include the framework into your project**: We intent to use the torch_geometric nn class to build the gnn and use the torch_geometric explain/profile class to show the results. A number of the other classes are also used to load the dataset. 

4. **What data are you going to run on (initially, may change)**: We are going to run on the Citeseer dataset that consists of 3,327 nodes, 9,104 edges, 3,703 features and 6 classes. There is only 1 graph in this network as all the nodes are connected to at least one other node. 

5. **What deep learning models do you expect to use**: We expect to use Graph neural network/Graph convolutional network.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
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
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

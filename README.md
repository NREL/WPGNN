## Wind Plant Graph Neural Network (WPGNN) and Plant Layout Generator (PLayGen)

This repository contains code for the model described in 

Harrison-Atlas, D., Glaws, A., King, R. N., and Lantz, E. "Artificial intelligence-aided wind plant optimization for nationwide evaluation of land use and economic benefits of wake steering." Nature Energy (2024). [https://doi.org/10.1038/s41560-024-01516-8](https://doi.org/10.1038/s41560-024-01516-8).

___

#### Description

The WPGNN is a graph neural network model for predicting wind plant performance. It represents the wind plant a graph with nodes representing individual turbines and wake effects encoded by directed edges. Global-, node-, and edge-level features of the graph represent atmospheric conditions, turbine locations and yaw angles, and the turbines' relative downwind positions, respectively. The model is contained in the `wpgnn.py` file. The `utils.py` file contains functionality to support data normalization and generating graph data structures. The `WPGNN_demo.ipynb` notebook provides a demonstration of how to train and evaluate the WPGNN. This demo operates on a small example dataset in `example_data.h5`. 

In addition to the WPGNN, we include the code for the plant layout generator `playgen.py`. This generator can produce random realizations of realistic wind plant layouts from one of the four canonical styles:  cluster, single string, multiple string, or parallel string. The `PLayGen_demo.ipynb` notebook provides a demonstration of how to use the generator tool.

A conda environment YML file `wpgnn_env.yml` has been provided for your convenience. To build this conda environment use the command

`conda env create -f wpgnn_env.yml`

and then

`conda activate wpgnn_env.yml`

#### Acknowledgments
This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

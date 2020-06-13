# Pytorch Template

## Introduction

Template to start developing models in Pytorch. What does the template do:

- Run training for your dataset

- Print metrics after each iteration

- Save model weights

- Load pretrained weights from file

- Save log file (.csv) of loss and metrics for train and validation data after each epoch

- Callbacks: Early_stopping and Reduce_learning_rate_on_plateau

- Save image (.png) of loss and learning curves after each epoch

## Usage

Things needed to be changed in a new project:

1. `train.py`: Change hyperparameters, data paths, loss function, and data transforms.

1. `dataset.py`: Define [PyTorch dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) which handles batch indices. Just inherit `Dataset`, and implement `__len__` and `__get__` member functions.

1. `network.py`: Add model structure.

1. `metrics.py`: Create metric functions that take only 2 inputs (prediction and ground truth). These functions are used automatically in the code.

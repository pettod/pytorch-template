# Pytorch Template

## Introduction

Template to start developing models in Pytorch. What does the template do:

- Run training for a dataset (training and validation datasets)

- Print metrics after each iteration

- Save model weights and optimizer state

- Load pretrained weights and optimizer state from file

- Callbacks: Early_stopping and Reduce_learning_rate_on_plateau

- Save log file (.csv) of loss and metrics for train and validation data after each epoch

- Save learning curves image (.png) of loss and metrics after each epoch

- Test model with validation data

## Usage

Things needed to be changed in a new project:

1. `train.py`: Change hyperparameters, data paths, loss function, and data transforms.

1. `src/dataset.py`: Define [PyTorch dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) which returns single sample with given sample index. Inherit `Dataset`, and implement `__len__` and `__getitem__` member functions.

1. `src/network.py`: Add model structure.

1. `src/loss_functions.py`: Define loss function and import it in `train.py`.

1. `src/metrics.py`: Create metric functions that take only 2 inputs (prediction and ground truth). These functions are used automatically in the code.

1. `test.py`: Modify tests depending on your prediction data type.

## Installation

You can define updated versions of the libraries for the project. Here in the example, no versions has been defined.

```bash
conda install pytorch torchvision
pip install -r requirements.txt
```

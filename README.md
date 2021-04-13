# Pytorch Template

## Introduction

Template to start developing models in Pytorch. What does the template do:

- Run training for a dataset (training and validation datasets)
- Print metrics after each iteration
- "Early stopping" callback to stop training when not progressing anymore
- Save one model or multiple models weights, optimizer(s) state, scheduler(s) state
- Save log file (.csv) of loss and metrics for train and validation data after each epoch
- Save learning curves image (.png) of loss and metrics after each epoch
- Load pretrained weights, optimizer state and scheduler state from file

## Usage

Things needed to be changed in a new project:

1. `config.py`: Change configurations of the training such as hyperparameters, data paths, loss function, optimizer, scheduler etc. 

1. `train.py`: Define data transforms and dataset calling.

1. `src/dataset.py`: Define [PyTorch dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) which returns single sample with given sample index. Inherit `Dataset`, and implement `__len__` and `__getitem__` member functions.

1. `src/learner.py`: If your data is more complex, implement `trainIteration` and `validationIteration` member functions.

1. `src/loss_functions.py`: Define loss function.

1. `src/metrics.py`: Create metric functions that take only 2 inputs (prediction and ground truth). These functions are used automatically in the code.

1. `src/network.py`: Add model structure.

1. `test.py`: Modify tests depending on your prediction data type.

## Installation

You can define updated versions of the libraries for the project. Here in the example, no versions has been defined.

```bash
conda install pytorch torchvision
pip install -r requirements.txt
```

## Run Training

```bash
python train.py
```

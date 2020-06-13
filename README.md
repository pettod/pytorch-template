# Pytorch Template

Template to start developing models in Pytorch

Things needed to be changed in a new project:

1. `train.py`: Hyperparameters, data paths, loss function, data transforms.

1. `network.py`: Model structure.

1. `metrics.py`: Metric functions that take only 2 inputs (prediction and ground truth). These functions are used automatically in the code.

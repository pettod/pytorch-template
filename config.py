import os
from multiprocessing import cpu_count

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.loss_functions import maeGradientPlusMae as lossFunction
from src.network import Net


DATA_ROOT = os.path.realpath("../REDS")


class CONFIG:
    # Paths
    TRAIN_X_DIR = os.path.join(DATA_ROOT, "train_blur/")
    TRAIN_Y_DIR = os.path.join(DATA_ROOT, "train_sharp/")
    VALID_X_DIR = os.path.join(DATA_ROOT, "val_blur/")
    VALID_Y_DIR = os.path.join(DATA_ROOT, "val_sharp/")

    # General parameters
    EPOCHS = 1000
    LOAD_MODEL = False
    MODEL_PATH = None
    DROP_LAST_BATCH = False
    NUMBER_OF_DATALOADER_WORKERS = cpu_count()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    LOSS_FUNCTION = lossFunction
    BATCH_SIZE = 16
    PATCH_SIZE = 256
    PATIENCE = 10
    LEARNING_RATE = 1e-4
    ITERATIONS_PER_EPOCH = 1

    # Model
    MODEL = Net()
    OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    SCHEDULER = ReduceLROnPlateau(
        OPTIMIZER, "min", 0.3, PATIENCE//3, min_lr=1e-8)

import os
from multiprocessing import cpu_count

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.network import Net


DATA_ROOT = os.path.realpath("../input/REDS")


class CONFIG:
    # Data paths
    TRAIN_X_DIR = os.path.join(DATA_ROOT, "train_blur/")
    TRAIN_Y_DIR = os.path.join(DATA_ROOT, "train_sharp/")
    VALID_X_DIR = os.path.join(DATA_ROOT, "val_blur/")
    VALID_Y_DIR = os.path.join(DATA_ROOT, "val_sharp/")
    TEST_IMAGE_PATH = None

    # Model loading
    LOAD_MODEL = False
    LOAD_OPTIMIZER_STATE = True
    CREATE_NEW_MODEL_PATH = True
    MODEL_PATH = None

    # Hyperparameters
    EPOCHS = 1000
    BATCH_SIZE = 16
    PATCH_SIZE = 256
    PATIENCE = 30
    ITERATIONS_PER_EPOCH = 1

    # Model
    MODELS = [
        Net(),
    ]
    OPTIMIZERS = [
        optim.Adam(MODELS[0].parameters(), lr=1e-4),
    ]
    SCHEDULERS = [
        ReduceLROnPlateau(OPTIMIZERS[0], "min", 0.3, 6, min_lr=1e-8),
    ]

    # General parameters
    DROP_LAST_BATCH = False
    NUMBER_OF_DATALOADER_WORKERS = cpu_count()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

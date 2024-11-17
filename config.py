import os
from multiprocessing import cpu_count

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import (
    Compose, CenterCrop, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip,
    ToTensor, Normalize
)

from src.dataset import ImageDataset as Dataset
from src.loss_functions import l1, sobelLoss, GANLoss
from src.architectures.discriminator import UNetDiscriminatorSN
from src.architectures.model import Net


DATA_ROOT = os.path.realpath("../input/REDS")


class CONFIG:
    # Data paths
    TRAIN_X_DIR = os.path.join(DATA_ROOT, "train_blur/")
    TRAIN_Y_DIR = os.path.join(DATA_ROOT, "train_sharp/")
    VALID_X_DIR = os.path.join(DATA_ROOT, "val_blur/")
    VALID_Y_DIR = os.path.join(DATA_ROOT, "val_sharp/")
    TEST_IMAGE_PATHS = [
    ]

    # Model
    SEED = 5432367
    MODELS = [
        Net(),
    ]
    OPTIMIZERS = [
        optim.Adam(MODELS[0].parameters(), lr=1e-4),
    ]
    SCHEDULERS = [
        ReduceLROnPlateau(OPTIMIZERS[0], "min", 0.3, 6, min_lr=1e-8),
    ]

    # Model loading
    LOAD_MODELS = [
        False,
    ]
    MODEL_PATHS = [
        None,
    ]
    LOAD_OPTIMIZER_STATES = [
        False,
    ]
    CREATE_NEW_MODEL_DIR = True

    # Cost function
    LOSS_FUNCTIONS = [
        [l1, sobelLoss],
    ]
    LOSS_WEIGHTS = [
        [1, 1],
    ]

    # GAN
    USE_GAN = False
    DISCRIMINATOR = UNetDiscriminatorSN(3)
    DIS_OPTIMIZER = optim.Adam(DISCRIMINATOR.parameters(), lr=1e-4)
    DIS_SCHEDULER = ReduceLROnPlateau(DIS_OPTIMIZER, "min", 0.3, 6, min_lr=1e-8)
    DIS_LOSS = GANLoss("vanilla")
    DIS_LOSS_WEIGHT = 1

    # Load GAN
    LOAD_GAN = False
    DIS_PATH = None
    LOAD_DIS_OPTIMIZER_STATE = False

    # Hyperparameters
    EPOCHS = 1000
    BATCH_SIZE = 16
    PATCH_SIZE = 256
    PATIENCE = 30
    ITERATIONS_PER_EPOCH = 10

    # Transforms and dataset
    TRAIN_TRANSFORM = Compose([
        RandomCrop(PATCH_SIZE),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])
    VALID_TRANSFORM = Compose([
        CenterCrop(PATCH_SIZE),
        ToTensor(),
    ])
    TEST_TRANSFORM = ToTensor()
    INPUT_NORMALIZE = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    TRAIN_DATASET = Dataset(TRAIN_X_DIR, TRAIN_Y_DIR, TRAIN_TRANSFORM, INPUT_NORMALIZE)
    VALID_DATASET = Dataset(VALID_X_DIR, VALID_Y_DIR, VALID_TRANSFORM, INPUT_NORMALIZE)

    # General parameters
    DROP_LAST_BATCH = False
    NUMBER_OF_DATALOADER_WORKERS = cpu_count()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

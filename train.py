import os
import torch
from torchvision import transforms

# Project files
from src.dataset import ImageDataset
from src.learner import Learner
from src.loss_functions import maeGradientPlusMae as lossFunction

# Data paths
DATA_ROOT = os.path.realpath("../../REDS")
TRAIN_X_DIR = os.path.join(DATA_ROOT, "train_blur/")
TRAIN_Y_DIR = os.path.join(DATA_ROOT, "train_sharp/")
VALID_X_DIR = os.path.join(DATA_ROOT, "val_blur/")
VALID_Y_DIR = os.path.join(DATA_ROOT, "val_sharp/")

# Model parameters
LOAD_MODEL = True
MODEL_PATH = None
BATCH_SIZE = 16
PATCH_SIZE = 256
PATIENCE = 10
LEARNING_RATE = 1e-4
DROP_LAST_BATCH = False
NUMBER_OF_DATALOADER_WORKERS = 8


def main():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(PATCH_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    valid_transforms = transforms.Compose([
        transforms.CenterCrop(PATCH_SIZE),
        transforms.ToTensor()
    ])
    train_dataset = ImageDataset(TRAIN_X_DIR, TRAIN_Y_DIR, train_transforms)
    valid_dataset = ImageDataset(VALID_X_DIR, VALID_Y_DIR, valid_transforms)
    learner = Learner(
        train_dataset, valid_dataset, BATCH_SIZE, LEARNING_RATE, lossFunction,
        PATIENCE, NUMBER_OF_DATALOADER_WORKERS, LOAD_MODEL, MODEL_PATH,
        DROP_LAST_BATCH)
    learner.train()


main()

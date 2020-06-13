import os
import torch
from torchvision import transforms

# Project files
from dataset import ImageDataset
from learner import Learner


# Data paths
DATA_ROOT = os.path.realpath("../../REDS")
TRAIN_X_DIR = os.path.join(DATA_ROOT, "train_blur/")
TRAIN_Y_DIR = os.path.join(DATA_ROOT, "train_sharp/")
VALID_X_DIR = os.path.join(DATA_ROOT, "val_blur/")
VALID_Y_DIR = os.path.join(DATA_ROOT, "val_sharp/")

# Model parameters
LOAD_MODEL = False
MODEL_PATH = None
BATCH_SIZE = 16
PATCH_SIZE = 256
PATIENCE = 10
LEARNING_RATE = 1e-4
DROP_LAST_BATCH = False


def lossFunction(y_pred, y_true):
    return torch.mean(y_pred - y_true)


def main():
    data_transforms = transforms.Compose([
        transforms.RandomCrop(PATCH_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    train_dataset = ImageDataset(TRAIN_X_DIR, TRAIN_Y_DIR, data_transforms)
    valid_dataset = ImageDataset(VALID_X_DIR, VALID_Y_DIR, data_transforms)
    learner = Learner(
        train_dataset, valid_dataset, data_transforms, BATCH_SIZE,
        LEARNING_RATE, lossFunction, PATIENCE, LOAD_MODEL, MODEL_PATH,
        DROP_LAST_BATCH)
    learner.train()


main()
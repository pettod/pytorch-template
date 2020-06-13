import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

import glob
import math
import numpy as np
import os
import time
from tqdm import trange

# Project files
from callbacks import CsvLogger, EarlyStopping
from network import Net
from dataset import ImageDataset
from utils import \
    getMetrics, getEmptyEpochMetrics, updateEpochMetrics, getProgressbarText, \
    plotLearningCurve


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


class Train():
    def __init__(self, data_transforms, drop_last_batch=False):
        # Device (CPU / CUDA)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: Running on CPU\n\n\n\n")

        # Model details and read metrics
        self.model_root = "saved_models"
        self.model = self.loadModel()
        save_model_directory = os.path.join(
            self.model_root, time.strftime("%Y-%m-%d_%H%M%S"))
        self.metrics = getMetrics()

        # Define optimizer and callbacks
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", 0.3, 3, min_lr=1e-8)
        self.csv_logger = CsvLogger(save_model_directory)
        self.early_stopping = EarlyStopping(save_model_directory, PATIENCE)
        self.epoch_metrics = getEmptyEpochMetrics()

        # Define train and validation batch generators
        train_dataset = ImageDataset(TRAIN_X_DIR, TRAIN_Y_DIR, data_transforms)
        valid_dataset = ImageDataset(VALID_X_DIR, VALID_Y_DIR, data_transforms)
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
            drop_last=drop_last_batch)
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
            drop_last=drop_last_batch)
        self.number_of_train_batches = len(train_dataset) / BATCH_SIZE
        self.number_of_valid_batches = len(valid_dataset) / BATCH_SIZE
        if drop_last_batch:
            self.number_of_train_batches = int(self.number_of_train_batches)
            self.number_of_valid_batches = int(self.number_of_valid_batches)
        else:
            self.number_of_train_batches = math.ceil(self.number_of_train_batches)
            self.number_of_valid_batches = math.ceil(self.number_of_valid_batches)

    def loadModel(self):
        if LOAD_MODEL:

            # Load latest model
            if MODEL_PATH is None:
                model_name = sorted(glob.glob(os.path.join(
                    self.model_root, *['*', "*.pt"])))[-1]
            else:

                # Load model based on index
                if type(MODEL_PATH) == int:
                    model_name = sorted(glob.glob(os.path.join(
                        self.model_root, *['*', "*.pt"])))[MODEL_PATH]

                # Load defined model path
                else:
                    model_name = MODEL_PATH
            model = nn.DataParallel(Net()).to(self.device)
            model.load_state_dict(torch.load(model_name))
            model.eval()
            print("Loaded model: {}".format(model_name))
        else:
            model = nn.DataParallel(Net()).to(self.device)
        print("{:,} model parameters".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))
        return model

    def lossFunction(self, y_pred, y_true):
        return torch.mean(y_pred - y_true)

    def validationRound(self):
        # Load tensor batch
        for i, (X, y) in zip(range(self.number_of_valid_batches), self.valid_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            output = self.model(X)
            loss = self.lossFunction(output, y)
            self.epoch_metrics["valid_loss"] += (
                loss.item() - self.epoch_metrics["valid_loss"]) / (i+1)
            updateEpochMetrics(
                output, y, i, self.epoch_metrics, "valid")
        validation_loss = self.epoch_metrics["valid_loss"]
        print("\n\n{}".format(getProgressbarText(
            self.epoch_metrics, "Valid")))
        self.csv_logger.__call__(self.epoch_metrics)
        self.early_stopping.__call__(validation_loss, self.model)
        self.scheduler.step(validation_loss)
        plotLearningCurve(model_root=self.model_root)

    def train(self):
        # Run epochs
        epochs = 1000
        for epoch in range(1, epochs+1):
            if self.early_stopping.isEarlyStop():
                print("Early stop")
                break
            progress_bar = trange(self.number_of_train_batches, leave=True)
            progress_bar.set_description(
                " Epoch {}/{}".format(epoch, epochs))
            self.epoch_metrics = getEmptyEpochMetrics()
            self.epoch_metrics["epoch"] = epoch

            # Run batches
            for i, (X, y) in zip(progress_bar, self.train_dataloader):

                # Run validation data before last batch
                if i == self.number_of_train_batches - 1:
                    with torch.no_grad():
                        self.validationRound()

                # Load tensor batch
                X = X.to(self.device)
                y = y.to(self.device)

                # Feed forward and backpropagation
                self.model.zero_grad()
                output = self.model(X)
                loss = self.lossFunction(output, y)
                loss.backward()
                self.optimizer.step()

                # Compute metrics
                with torch.no_grad():
                    self.epoch_metrics["train_loss"] += (
                        loss.item() - self.epoch_metrics["train_loss"]) / (i+1)
                    updateEpochMetrics(
                        output, y, i, self.epoch_metrics, "train")
                    progress_bar.display(
                        getProgressbarText(self.epoch_metrics, "Train"), 1)


def main():
    data_transforms = transforms.Compose([
        transforms.RandomCrop(PATCH_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    train = Train(data_transforms)
    train.train()


main()

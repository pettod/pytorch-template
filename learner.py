import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import os
import time
from tqdm import trange

# Project files
from callbacks import CsvLogger, EarlyStopping
from utils import \
    getMetrics, getEmptyEpochMetrics, updateEpochMetrics, getProgressbarText, \
    saveLearningCurve, loadModel, numberOfDatasetBatches


class Learner():
    def __init__(
            self, train_dataset, valid_dataset, data_transforms, batch_size,
            learning_rate, loss_function, patience, num_workers=1,
            load_pretrained_weights=False, model_path=None,
            drop_last_batch=False):
        # Device (CPU / CUDA)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: Running on CPU\n\n\n\n")

        # Model details
        self.model_root = "saved_models"
        self.model = loadModel(
            self.model_root, load_pretrained_weights, model_path
            ).to(self.device)
        save_model_directory = os.path.join(
            self.model_root, time.strftime("%Y-%m-%d_%H%M%S"))

        # Define optimizer and callbacks
        self.loss_function = loss_function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", 0.3, 3, min_lr=1e-8)
        self.csv_logger = CsvLogger(save_model_directory)
        self.early_stopping = EarlyStopping(save_model_directory, patience)
        self.epoch_metrics = getEmptyEpochMetrics()

        # Define train and validation batch generators
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=drop_last_batch)
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=drop_last_batch)
        self.number_of_train_batches = numberOfDatasetBatches(
            train_dataset, batch_size, drop_last_batch)
        self.number_of_valid_batches = numberOfDatasetBatches(
            valid_dataset, batch_size, drop_last_batch)

    def validationEpoch(self):
        # Load tensor batch
        for i, (X, y) in zip(
                range(self.number_of_valid_batches), self.valid_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            output = self.model(X)
            loss = self.loss_function(output, y)
            self.epoch_metrics["valid_loss"] += (
                loss.item() - self.epoch_metrics["valid_loss"]) / (i+1)
            updateEpochMetrics(
                output, y, i, self.epoch_metrics, "valid")
        validation_loss = self.epoch_metrics["valid_loss"]
        print("\n\n{}".format(getProgressbarText(self.epoch_metrics, "Valid")))
        self.csv_logger.__call__(self.epoch_metrics)
        self.early_stopping.__call__(validation_loss, self.model)
        self.scheduler.step(validation_loss)
        saveLearningCurve(model_root=self.model_root)

    def train(self):
        # Run epochs
        epochs = 1000
        for epoch in range(1, epochs+1):
            if self.early_stopping.isEarlyStop():
                print("Early stop")
                break
            progress_bar = trange(self.number_of_train_batches, leave=True)
            progress_bar.set_description(" Epoch {}/{}".format(epoch, epochs))
            self.epoch_metrics = getEmptyEpochMetrics()
            self.epoch_metrics["epoch"] = epoch

            # Run batches
            for i, (X, y) in zip(progress_bar, self.train_dataloader):

                # Run validation data before last batch
                if i == self.number_of_train_batches - 1:
                    with torch.no_grad():
                        self.validationEpoch()

                # Feed forward and backpropagation
                X, y = X.to(self.device), y.to(self.device)
                self.model.zero_grad()
                output = self.model(X)
                loss = self.loss_function(output, y)
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

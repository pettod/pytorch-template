import os
from glob import glob
from shutil import copy, move, rmtree

import numpy as np
import pandas as pd
import torch


def createSaveModelDirectory(save_directory):
    # Create folders if do not exist
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a 
    given patience.
    """
    def __init__(
            self, save_model_directory, patience=10, verbose=False, delta=0,
            counter=0, validation_loss_min=np.Inf):
        """
        Args:
            patience : int
                How long to wait after last time validation loss improved.
            verbose : bool
                If True, prints a message for each validation loss improvement. 
            delta : float
                Minimum change in the monitored quantity to qualify as an
                improvement.
            counter : int
                Early stopping counter
            validation_loss_min : float
                Minimum validation loss achieved
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.valid_loss_min = validation_loss_min
        self.delta = delta
        self.save_directory = save_model_directory
        self.tmp_folder = os.path.join(
            "tmp", self.save_directory.split(os.sep)[-1])
        self.saved_files = [
            "config.py",
            "src/network.py",
            "src/loss_functions.py",
        ]
        self.saveNetworkConfig()

    def saveNetworkConfig(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        for file_path in self.saved_files:
            copy(file_path, self.tmp_folder)

    def moveNetworkConfigToSaveDirectory(self):
        # 'tmp/' folder exists
        if os.path.isdir(self.tmp_folder):

            # Loop each file in 'tmp/' folder
            for file_path in glob(os.path.join(self.tmp_folder, '*')):

                # Move file to 'save_models/' folder if not yet there
                if not os.path.isfile(os.path.join(
                        self.save_directory, os.path.basename(file_path))):
                    move(file_path, self.save_directory)
        
            # Remove 'tmp/' folder (even with files inside)
            rmtree(self.tmp_folder, ignore_errors=True)

    def __call__(self, epoch_metrics, model, optimizer):
        createSaveModelDirectory(self.save_directory)
        self.moveNetworkConfigToSaveDirectory()
        valid_loss = epoch_metrics["valid_loss"]
        score = -valid_loss
        if self.best_score is None:
            self.best_score = score
            self.saveCheckpoint(epoch_metrics, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.saveCheckpoint(epoch_metrics, model, optimizer)
            self.counter = 0

    def saveCheckpoint(self, epoch_metrics, model, optimizer):
        """Saves model when validation loss decrease."""
        if type(model) != list:
            model = [model]
            optimizer = [optimizer]
        models_state_dict = {}
        for i in range(len(model)):
            models_state_dict[f"model_{i}"] = model[i].state_dict()
            models_state_dict[f"optimizer_{i}"] = optimizer[i].state_dict()
        torch.save({
            **models_state_dict,
            **epoch_metrics},
            os.path.join(self.save_directory, "model.pt")
        )
        if self.verbose:
            print("Validation loss decreased. Model saved")
        self.valid_loss_min = epoch_metrics["valid_loss"]

    def isEarlyStop(self):
        if self.early_stop:
            print("Early stop")
        return self.early_stop


class CsvLogger:
    def __init__(self, save_model_directory):
        self.save_directory = save_model_directory
        self.logs_file_path = os.path.join(self.save_directory, "logs.csv")

    def __call__(self, loss_and_metrics):
        createSaveModelDirectory(self.save_directory)

        # Create CSV file
        new_data_frame = pd.DataFrame(loss_and_metrics, index=[0])
        if not os.path.isfile(self.logs_file_path):
            new_data_frame.to_csv(
                self.logs_file_path, header=True, index=False)
        else:
            with open(self.logs_file_path, 'a') as old_data_frame:
                new_data_frame.to_csv(
                    old_data_frame, header=False, index=False)
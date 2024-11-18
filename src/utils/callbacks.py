import os
from glob import glob
from shutil import copy, move, rmtree

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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
            "src/architectures",
            "src/dataset.py",
            "src/loss_functions.py",
        ]
        self.saveNetworkConfig()

    def saveNetworkConfig(self):
        code_file_save_folder = os.path.join(self.tmp_folder, "codes")
        os.makedirs(code_file_save_folder, exist_ok=True)
        for file_path in self.saved_files:
            if os.path.isdir(file_path):
                folder_file_paths = glob(os.path.join(file_path, "*.py"))
                for folder_file_path in folder_file_paths:
                    copy(folder_file_path, code_file_save_folder)
            else:
                copy(file_path, code_file_save_folder)

    def createSaveModelDirectory(self):
        # Create folders if do not exist
        if not os.path.isdir(self.save_directory):
            os.makedirs(self.save_directory)
            self.moveNetworkConfigToSaveDirectory()

    def moveNetworkConfigToSaveDirectory(self):
        # 'tmp/' folder exists but save directory doesn't
        if os.path.isdir(self.tmp_folder):

            # Loop each file in 'tmp/' folder
            for file_path in glob(os.path.join(self.tmp_folder, '*')):

                # Move file to 'save_models/' folder if not yet there
                if not os.path.isfile(os.path.join(
                        self.save_directory, os.path.basename(file_path))):
                    move(file_path, self.save_directory)
        
            # Remove 'tmp/' folder (even with files inside)
            rmtree(self.tmp_folder, ignore_errors=True)

    def __call__(
            self, epoch_metrics, model, optimizer, last_discriminator=False):
        self.createSaveModelDirectory()
        valid_loss = epoch_metrics["valid_total-loss"]
        score = -valid_loss
        if self.best_score is None:
            self.best_score = score
            self.saveCheckpoint(
                epoch_metrics, model, optimizer, last_discriminator)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.saveCheckpoint(
                epoch_metrics, model, optimizer, last_discriminator)
            self.counter = 0
        self.saveCheckpoint(
            epoch_metrics, model, optimizer, last_discriminator, True)

    def saveCheckpoint(
            self, epoch_metrics, model, optimizer, last_discriminator,
            save_last_epoch_model=False):
        """Saves model when validation loss decrease."""
        if type(model) != list:
            model = [model]
            optimizer = [optimizer]
        models_state_dict = {}
        for i in range(len(model)):
            if i == len(model) - 1 and last_discriminator:
                model_name = "discriminator.pt"
            else:
                model_name = f"model_{i}.pt"
            if save_last_epoch_model:
                model_name = f"last_epoch_{model_name}"
            if type(model[i]) == nn.DataParallel:
                saved_model = model[i].module
            else:
                saved_model = model[i]
            torch.save(
                saved_model.state_dict(),
                os.path.join(self.save_directory, model_name))
            models_state_dict[f"optimizer_{i}"] = optimizer[i].state_dict()
        checkpoint_name = "checkpoint.ckpt"
        if save_last_epoch_model:
            checkpoint_name = f"last_epoch_{checkpoint_name}"
        else:
            if self.verbose:
                print("Validation loss decreased. Model saved")
            self.valid_loss_min = epoch_metrics["valid_total-loss"]
        torch.save({
            **models_state_dict,
            **epoch_metrics},
            os.path.join(self.save_directory, checkpoint_name)
        )

    def isEarlyStop(self):
        if self.early_stop:
            print("Early stop")
        return self.early_stop


class CsvLogger():
    def __init__(self, save_model_directory):
        self.logs_file_path = os.path.join(save_model_directory, "logs.csv")

    def __call__(self, loss_and_metrics):
        # Create CSV file
        new_data_frame = pd.DataFrame(loss_and_metrics, index=[0])
        if not os.path.isfile(self.logs_file_path):
            new_data_frame.to_csv(
                self.logs_file_path, header=True, index=False)
        else:
            with open(self.logs_file_path, 'a') as old_data_frame:
                new_data_frame.to_csv(
                    old_data_frame, header=False, index=False)

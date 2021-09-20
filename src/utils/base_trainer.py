import torch
import torch.nn as nn
from tqdm import trange

from config import CONFIG
import src.utils.callbacks as cb
import src.utils.utils as ut


class Basetrainer():
    def __init__(self, train_dataset, valid_dataset):
        self.epoch_metrics = {}

        # Load from config
        self.epochs = CONFIG.EPOCHS
        self.model = nn.DataParallel(CONFIG.MODEL).to(CONFIG.DEVICE)
        self.optimizer = CONFIG.OPTIMIZER
        self.scheduler = CONFIG.SCHEDULER
        self.loss_function = CONFIG.LOSS_FUNCTION

        # Callbacks
        self.start_epoch, self.model_directory, validation_loss_min = \
            ut.loadModel(
                self.model, self.epoch_metrics, CONFIG.MODEL_PATH,
                self.optimizer, CONFIG.LOAD_MODEL)
        self.csv_logger = cb.CsvLogger(self.model_directory)
        self.early_stopping = cb.EarlyStopping(
            self.model_directory, CONFIG.PATIENCE,
            validation_loss_min=validation_loss_min)

        # Train and validation batch generators
        self.train_dataloader = ut.getDataloader(train_dataset)
        self.valid_dataloader = ut.getDataloader(valid_dataset, shuffle=False)
        self.number_of_train_batches = ut.getIterations(self.train_dataloader)
        self.number_of_valid_batches = len(self.valid_dataloader)

    def logData(self):
        self.csv_logger.__call__(self.epoch_metrics)
        self.early_stopping.__call__(
            self.epoch_metrics, self.model, self.optimizer)
        self.scheduler.step(self.epoch_metrics["valid_loss"])
        ut.saveLearningCurve(model_directory=self.model_directory)

    def validationEpoch(self):
        print()
        progress_bar = trange(self.number_of_valid_batches, leave=False)
        progress_bar.set_description(" Validation")
        for i, batch in zip(progress_bar, self.valid_dataloader):
            prediction, y, loss = self.validationIteration(batch)
            ut.updateEpochMetrics(
                prediction, y, loss, i, self.epoch_metrics, "valid",
                self.optimizer)
        print("\n{}".format(ut.getProgressbarText(
            self.epoch_metrics, "Valid")))
        self.logData()

    def trainEpoch(self, epoch):
        progress_bar = trange(self.number_of_train_batches, leave=False)
        progress_bar.set_description(f" Epoch {epoch}/{self.epochs}")
        for i, batch in zip(progress_bar, self.train_dataloader):

            # Validation epoch before last batch
            if i == self.number_of_train_batches - 1:
                with torch.no_grad():
                    self.validationEpoch()
            prediction, y, loss = self.trainIteration(batch)

            # Compute metrics, print progress bar
            with torch.no_grad():
                ut.updateEpochMetrics(
                    prediction, y, loss, i, self.epoch_metrics, "train")
                progress_bar.display(
                    ut.getProgressbarText(self.epoch_metrics, "Train"), 1)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            if self.early_stopping.isEarlyStop():
                break
            self.epoch_metrics = ut.initializeEpochMetrics(epoch)
            self.trainEpoch(epoch)

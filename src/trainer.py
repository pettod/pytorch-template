from posixpath import basename
import torch

import src.utils.utils as ut
from src.utils.base_trainer import Basetrainer


class Trainer(Basetrainer):
    def forwardPropagation(self, batch):
        x, y = ut.toDevice(batch)
        prediction = self.model(x)
        loss = self.loss_function(prediction, y)
        return y, prediction, loss

    def logData(self):
        self.csv_logger.__call__(self.epoch_metrics)
        self.early_stopping.__call__(
            self.epoch_metrics, self.model, self.optimizer)
        self.scheduler.step(self.epoch_metrics["valid_loss"])
        ut.saveLearningCurve(model_directory=self.model_directory)

    def validationIteration(self, batch, i):
        y, prediction, loss = self.forwardPropagation(batch)
        ut.updateEpochMetrics(
            prediction, y, loss, i, self.epoch_metrics, "valid",
            self.optimizer)

    def trainIteration(self, batch, i):
        y, prediction, loss = self.forwardPropagation(batch)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            ut.updateEpochMetrics(
                prediction, y, loss, i, self.epoch_metrics, "train")

import os
import torch
import torch.nn as nn
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
import src.utils.callbacks as cb
import src.utils.utils as ut


class Basetrainer():
    def __init__(self, train_dataset, valid_dataset):
        self.epoch_metrics = {}
        self.iteration_losses = {}

        # Load from config
        self.epochs = CONFIG.EPOCHS
        self.models = [
            nn.DataParallel(m).to(CONFIG.DEVICE) for m in CONFIG.MODELS]
        self.optimizers = CONFIG.OPTIMIZERS
        self.schedulers = CONFIG.SCHEDULERS
        self.loss_functions = CONFIG.LOSS_FUNCTIONS
        self.loss_weights = CONFIG.LOSS_WEIGHTS
        self.use_gan = CONFIG.USE_GAN
        if self.use_gan:
            self.discriminator = nn.DataParallel(
                CONFIG.DISCRIMINATOR).to(CONFIG.DEVICE)
            self.discriminator = CONFIG.DISCRIMINATOR
            self.dis_optimizer = CONFIG.DIS_OPTIMIZER
            self.dis_scheduler = CONFIG.DIS_SCHEDULER
            self.dis_loss = CONFIG.DIS_LOSS
            self.dis_loss_weight = CONFIG.DIS_LOSS_WEIGHT

            # Load discriminator
            ut.loadModel(
                self.discriminator, CONFIG.LOAD_GAN, CONFIG.DIS_PATH,
                "discriminator.pt", self.dis_optimizer,
                CONFIG.LOAD_DIS_OPTIMIZER_STATE)

        # Load models
        for i in range(len(CONFIG.MODELS)):
            self.start_epoch, self.model_directory, validation_loss_min = \
                ut.loadModel(
                    self.models[i], CONFIG.LOAD_MODELS[i],
                    CONFIG.MODEL_PATHS[i], f"model_{i}.pt", self.optimizers[i],
                    CONFIG.LOAD_OPTIMIZER_STATES[i],
                    create_new_model_directory=CONFIG.CREATE_NEW_MODEL_DIR)

        # Callbacks
        self.csv_logger = cb.CsvLogger(self.model_directory)
        self.early_stopping = cb.EarlyStopping(
            self.model_directory, CONFIG.PATIENCE,
            validation_loss_min=validation_loss_min)
        self.tensorboard_writer = None

        # Train and validation batch generators
        self.train_dataloader = ut.getDataloader(train_dataset)
        self.valid_dataloader = ut.getDataloader(valid_dataset, shuffle=False)
        self.number_of_train_batches = ut.getIterations(self.train_dataloader)
        self.number_of_valid_batches = len(self.valid_dataloader)

    def costFunction(self, prediction, y, model_index):
        losses = []
        mi = model_index
        for l, w in zip(self.loss_functions[mi], self.loss_weights[mi]):
            loss = l(prediction, y) * w
            losses.append(loss)
            if str(type(l)) == "<class 'function'>":
                loss_name = l.__name__
            else:
                loss_name = l.__class__.__name__

            # Only last model doesn't have index in the losses
            if model_index != len(self.models) - 1:
                loss_name += f"-{model_index}"
            self.iteration_losses[loss_name] = loss

        # Use GAN if defined in CONFIG and last model in use
        if self.use_gan and model_index == len(self.models) - 1:
            self.discriminator.zero_grad()
            gen_dis_prediction = self.discriminator(prediction)
            gen_gan_loss = self.dis_loss(gen_dis_prediction, True)
            losses.append(gen_gan_loss)
            self.iteration_losses["gen-gan-loss"] = gen_gan_loss
        total_loss = sum(losses)
        total_loss_name = "total-loss"

        # Only last model doesn't have index in the losses
        if model_index != len(self.models) - 1:
            total_loss_name += f"-{model_index}"

        self.iteration_losses[total_loss_name] = total_loss
        return total_loss

    def discriminatorLoss(self, prediction, y):
        gen_dis_prediction = self.discriminator(prediction.detach())
        true_dis_prediction = self.discriminator(y)
        dis_gan_loss = \
            self.dis_loss(true_dis_prediction, True) + \
            self.dis_loss(gen_dis_prediction, False)
        return dis_gan_loss

    def trainDiscriminator(self, prediction, y):
        self.discriminator.zero_grad()
        dis_gan_loss = self.discriminatorLoss(prediction, y)
        dis_gan_loss.backward()
        self.dis_optimizer.step()
        self.iteration_losses["dis-gan-loss"] = dis_gan_loss

    def logData(self):
        self.csv_logger.__call__(self.epoch_metrics)
        if self.use_gan:
            self.early_stopping.__call__(
                self.epoch_metrics, self.models + [self.discriminator],
                self.optimizers + [self.dis_optimizer], True)
        else:
            self.early_stopping.__call__(
                self.epoch_metrics, self.models, self.optimizers, False)
        for s in self.schedulers:
            s.step(self.epoch_metrics["valid_total-loss"])
        if self.use_gan:
            self.dis_scheduler.step(self.epoch_metrics["valid_dis-gan-loss"])
        ut.saveLearningCurve(model_directory=self.model_directory)

        # Update Tensorboard
        if self.tensorboard_writer is None:
            self.tensorboard_writer = SummaryWriter(self.model_directory)
        for key, value in self.epoch_metrics.items():
            if key != "epoch":
                loop_metric = key.split('_')
                self.tensorboard_writer.add_scalar(
                    "{}/{}".format(loop_metric[1], loop_metric[0]),
                    value, self.epoch_metrics["epoch"])

    def validationEpoch(self):
        print()
        progress_bar = trange(self.number_of_valid_batches, leave=False)
        progress_bar.set_description(" Validation")
        for i, batch in zip(progress_bar, self.valid_dataloader):
            prediction, y = self.validationIteration(batch)
            if self.use_gan:
                dis_gan_loss = self.discriminatorLoss(prediction, y)
                self.iteration_losses["dis-gan-loss"] = dis_gan_loss
            ut.updateEpochMetrics(
                prediction, y, self.iteration_losses, i, self.epoch_metrics,
                "valid", self.optimizers)
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
            prediction, y = self.trainIteration(batch)
            if self.use_gan:
                self.trainDiscriminator(prediction, y)

            # Compute metrics, print progress bar
            with torch.no_grad():
                ut.updateEpochMetrics(
                    prediction, y, self.iteration_losses, i,
                    self.epoch_metrics, "train")
                progress_bar.display(
                    ut.getProgressbarText(self.epoch_metrics, "Train"), 1)

    def testModel(self, epoch):
        if CONFIG.TEST_IMAGE_PATH is not None:
            with torch.no_grad():
                test_image = self.testAfterEpoch()
            image_name = os.path.basename(CONFIG.TEST_IMAGE_PATH).split('.')[0]
            save_path = os.path.join(
                self.model_directory,
                "test_images",
                f"epoch_{epoch}_{image_name}.png")
            ut.saveTensorImage(test_image, save_path)
            self.tensorboard_writer.add_image("test_image", test_image, epoch)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            if self.early_stopping.isEarlyStop():
                break
            self.epoch_metrics = ut.initializeEpochMetrics(epoch)
            self.trainEpoch(epoch)
            self.testModel(epoch)

import cv2
import numpy as np
import os
import time
import torch
import torch.nn as nn
from imco import compareImages
from importlib import import_module
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project files
from src.dataset import ImageDataset
from src.utils.utils import loadModel

# Data paths
DATA_ROOT = os.path.realpath("../input/REDS")
VALID_X_DIR = os.path.join(DATA_ROOT, "val_blur/")
VALID_Y_DIR = os.path.join(DATA_ROOT, "val_sharp/")

# Model parameters
MODEL_PATHS = [
    "saved_models/2021-10-10_213030",
    "saved_models/2021-10-10_213030",
]
NAMES = [
    "Input",
    "model 1",
    "model 2",
    "Ground truth",
]
PATCH_SIZE = 256
DEVICE = torch.device("cuda")


def main():
    # Dataset
    valid_transforms = transforms.Compose([
        transforms.CenterCrop(PATCH_SIZE),
        transforms.ToTensor(),
    ])
    valid_dataset = ImageDataset(VALID_X_DIR, VALID_Y_DIR, valid_transforms)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=4, shuffle=False, num_workers=8)

    # Save directory
    save_directory = os.path.join(
        "predictions", time.strftime("%Y-%m-%d_%H%M%S"))
    with torch.no_grad():

        # Load models
        models = []
        for model_path in MODEL_PATHS:
            config = import_module(os.path.join(
                model_path, "config").replace('/', '.')).CONFIG
            model = nn.DataParallel(config.MODELS[0]).to(DEVICE)
            loadModel(model, model_path=model_path)
            models.append(model)

        # Predict and save
        for i, (x, y) in enumerate(tqdm(valid_dataloader)):
            x, y = x.to(DEVICE), y.numpy()
            predictions = [m(x).cpu().numpy() for m in models]
            x += 1
            x /= 2
            x = x.cpu().numpy()
            for j in range(x.shape[0]):
                images = [x[j]] + [p[j] for p in predictions] + [y[j]]
                images = [(255 * np.moveaxis(img, 0, -1)).astype(np.uint8) for img in images]
                concat_image = compareImages(images, NAMES, True).astype(np.uint8)
                if not os.path.isdir(save_directory):
                    os.makedirs(save_directory)
                cv2.imwrite(os.path.join(
                    save_directory, f"{i}_{j}.png"), cv2.cvtColor(
                        concat_image, cv2.COLOR_RGB2BGR))


main()

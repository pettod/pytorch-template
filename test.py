import cv2
import numpy as np
import os
import time
import torch
import torch.nn as nn
from imco import compareImages
from importlib import import_module
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project files
from src.dataset import ImageDataset as Dataset
from src.utils.utils import loadModel

# Data paths
DATA_ROOT = os.path.realpath("../input/REDS")
X_DIR = os.path.join(DATA_ROOT, "valid_blur")
Y_DIR = os.path.join(DATA_ROOT, "valid_sharp")

# Model parameters
MODEL_PATHS = [
    "saved_models/2022-05-20_212932",
    "saved_models/2022-05-20_212932",
]
NAMES = [
    "Input",
    "model 1",
    "model 2",
    "Ground truth",
]
PATCH_SIZE = 256
DEVICE = torch.device("cpu")


def loadModels():
    models = []
    for model_path in MODEL_PATHS:
        config = import_module(os.path.join(
            model_path, "codes.config").replace('/', '.')).CONFIG
        model = config.MODELS[0]
        loadModel(model, model_path=model_path)
        models.append(model.to(DEVICE))
    return models


def saveImage(save_directory, image, image_name):
    os.makedirs(save_directory, exist_ok=True)
    cv2.imwrite(os.path.join(save_directory, image_name), image[..., ::-1])


def main():
    # Dataset
    transforms = Compose([
        CenterCrop(PATCH_SIZE),
        ToTensor(),
    ])
    input_normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    dataset = Dataset(X_DIR, Y_DIR, transforms, input_normalize)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)

    # Save directory
    save_directory = os.path.join("predictions", time.strftime("%Y-%m-%d_%H%M%S"))
    with torch.no_grad():
        models = loadModels()

        # Predict and save
        for i, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(DEVICE), y.numpy()
            predictions = [m(x).cpu().numpy() for m in models]
            x += 1
            x /= 2
            x = x.cpu().numpy()
            for j in range(x.shape[0]):
                images = [x[j]] + [p[j] for p in predictions] + [y[j]]
                images = [(255 * np.moveaxis(img, 0, -1)).astype(np.uint8) for img in images]
                concat_image = compareImages(images, NAMES, True).astype(np.uint8)
                saveImage(save_directory, concat_image, f"{i}_{j}.png")


main()

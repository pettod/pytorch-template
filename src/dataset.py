import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def readImage(image_path):
    return Image.open(image_path).convert("RGB")


def readImagePaths(data_path):
    return np.array(sorted(glob(f"{data_path}/*/*.png")))


class ImageDataset(Dataset):
    def __init__(self, input_path, target_path=None, transform=None):
        self.input_image_paths = readImagePaths(input_path)
        self.target_image_paths = target_path
        if target_path:
            self.target_image_paths = readImagePaths(target_path)
        self.transform = transform
        self.input_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, sample_index):
        seed = random.randint(0, 2**32)
        input_image = readImage(self.input_image_paths[sample_index])
        if self.transform:
            random.seed(seed)  # Pytorch <= 1.5.1
            torch.manual_seed(seed)  # Pytorch 1.6.0
            input_image = self.input_normalize(self.transform(input_image))
        if self.target_image_paths is not None:
            target_image = readImage(self.target_image_paths[sample_index])
            if self.transform:
                random.seed(seed)
                torch.manual_seed(seed)
                target_image = self.transform(target_image)
            return input_image, target_image
        return input_image

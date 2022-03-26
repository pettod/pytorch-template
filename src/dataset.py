import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def readImage(image_path, transform, seed):
    image = Image.open(image_path).convert("RGB")
    if transform:
        random.seed(seed)  # Pytorch <= 1.5.1
        torch.manual_seed(seed)  # Pytorch 1.6.0
        image = transform(image)
    return image


def readImagePaths(data_path):
    if data_path is None or type(data_path) == list:
        return data_path
    return np.array(sorted(glob(f"{data_path}/*/*.png")))


class ImageDataset(Dataset):
    def __init__(
            self, input_path, target_path=None, transform=None,
            input_normalize=None):
        self.input_image_paths = readImagePaths(input_path)
        self.target_image_paths = readImagePaths(target_path)
        self.transform = transform
        self.input_normalize = input_normalize

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, i):
        seed = random.randint(0, 2**32)
        input_image = readImage(self.input_image_paths[i], self.transform, seed)
        if self.input_normalize:
            input_image = self.input_normalize(input_image)
        if self.target_image_paths is not None:
            target_image = readImage(
                self.target_image_paths[i], self.transform, seed)
            return input_image, target_image
        return input_image, self.input_image_paths[i]

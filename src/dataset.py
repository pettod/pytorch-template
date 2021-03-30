import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, input_path, target_path, transform=None):
        self.input_image_paths = np.array(sorted(glob(f"{input_path}/*/*.png")))
        self.target_image_paths = np.array(sorted(glob(f"{target_path}/*/*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.target_image_paths)

    def __getitem__(self, sample_index):
        input_image = Image.open(self.input_image_paths[sample_index])
        target_image = Image.open(self.target_image_paths[sample_index])
        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            input_image = self.transform(input_image)
            random.seed(seed)
            target_image = self.transform(target_image)
        return np.array(input_image), np.array(target_image)

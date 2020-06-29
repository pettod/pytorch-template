import cv2
from glob import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np


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


if __name__ == "__main__":
    DATA_ROOT = os.path.realpath("../../REDS")
    TRAIN_X_DIR = os.path.join(DATA_ROOT, "train_blur/")
    TRAIN_Y_DIR = os.path.join(DATA_ROOT, "train_sharp/")
    PATCH_SIZE = 256
    BATCH_SIZE = 4
    data_transform = transforms.Compose([
        transforms.RandomCrop(PATCH_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    dataset = ImageDataset(TRAIN_X_DIR, TRAIN_Y_DIR, transform=data_transform)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    for batch in dataloader:
        x, y = batch
        x, y = x.numpy(), y.numpy()
        print(x.shape)
        cv2.imshow("input, ground truth", cv2.cvtColor(cv2.hconcat(
            [x[0], y[0]]), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

from torchvision import transforms

# Project files
from config import CONFIG
from src.dataset import ImageDataset as Dataset
from src.learner import Learner


def main():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(CONFIG.PATCH_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])
    valid_transforms = transforms.Compose([
        transforms.CenterCrop(CONFIG.PATCH_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = Dataset(
        CONFIG.TRAIN_X_DIR, CONFIG.TRAIN_Y_DIR, train_transforms)
    valid_dataset = Dataset(
        CONFIG.VALID_X_DIR, CONFIG.VALID_Y_DIR, valid_transforms)
    learner = Learner(train_dataset, valid_dataset)
    learner.train()


main()

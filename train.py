from torchvision import transforms

from config import CONFIG
from src.dataset import ImageDataset as Dataset
from src.learner import Learner


def main():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(CONFIG.PATCH_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    valid_transforms = transforms.Compose([
        transforms.CenterCrop(CONFIG.PATCH_SIZE),
        transforms.ToTensor(),
    ])
    train_dataset = Dataset(
        CONFIG.TRAIN_X_DIR, CONFIG.TRAIN_Y_DIR, train_transforms)
    valid_dataset = Dataset(
        CONFIG.VALID_X_DIR, CONFIG.VALID_Y_DIR, valid_transforms)
    learner = Learner(train_dataset, valid_dataset)
    learner.train()


main()

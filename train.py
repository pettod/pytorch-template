from config import CONFIG
from src.trainer import Trainer


def main():
    trainer = Trainer(CONFIG.TRAIN_DATASET, CONFIG.VALID_DATASET)
    trainer.train()


main()

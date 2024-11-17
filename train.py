import optuna

from config import CONFIG
from src.optuna_objective import Objective
from src.trainer import Trainer
from src.utils.utils import seedEverything


TRAINING_PARAMETERS = [
    {"accuracy": 0.99, "flops": 100000, "loss": 2.0, "seed": 24132423, "lr": 0.04, "epochs": 100, "features": 4, "conv_layers": 4},
]
USED_PARAMETERS = None #TRAINING_PARAMETERS[-1]


def main():
    if USED_PARAMETERS:
        obj = Objective()
        obj(optuna.trial.FixedTrial(USED_PARAMETERS), save_model=True)
    else:
        if CONFIG.SEED:
            seedEverything(CONFIG.SEED)
        trainer = Trainer(CONFIG)
        trainer.train()


if __name__ == "__main__":
    main()

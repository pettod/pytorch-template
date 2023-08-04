import optuna
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import CONFIG
from src.trainer import Trainer
from src.architectures.model import Net
from flops import flops
from src.utils.utils import seedEverything


class Objective:
    """
    Optuna objetive class
    """
    def __init__(self):
        pass

    def __call__(self, trial: optuna.trial.Trial, save_model=False):
        """
        Objective function for optuna
        Chooses a model and hyperparameters and trains it on the training data
        
        Args:
            trial (optuna.trial.Trial): optuna trial object
        Returns:
            float: R^2 score of model on validation set
        """
        seed = trial.suggest_int("seed", 1, 2**32 - 1)
        seedEverything(seed)

        kwargs={
            #"loss_name": trial.suggest_categorical("loss_name", ["mae", "mse"]),
            #"optim_name": trial.suggest_categorical("optim_name", ["Bop", "Adam"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-4), # 1e-5, 1.
            #"learning_decay": trial.suggest_loguniform("bnn_weight_decay", 1e-5, 1.),
            #"scheduler_step_size": trial.suggest_int("bnn_scheduler_step_size", 1, 100),
            #"scheduler_gamma": trial.suggest_loguniform("bnn_scheduler_gamma", 1e-8, 1.),
            #"batch_size": trial.suggest_int("batch_size", 1, 100),
            "kernel_size": trial.suggest_int("kernel_size", 3, 3),
            "epochs": trial.suggest_int("epochs", 200, 300),
            "features": trial.suggest_int("features", 1, 8),
            "conv_layers": trial.suggest_int("conv_layers", 4, 4),
        }
        # Apply parameters and train a model
        cfg = CONFIG()
        cfg.MODELS[0] = Net(
            kwargs["kernel_size"],
            kwargs["features"],
            kwargs["conv_layers"],
            hist_size=256,
            output_size=4,
            input_halvings=kwargs["input_halvings"],
        )
        cfg.OPTIMIZERS[0] = optim.Adam(cfg.MODELS[0].parameters(), lr=kwargs["learning_rate"])
        cfg.SCHEDULERS[0] = ReduceLROnPlateau(cfg.OPTIMIZERS[0], "min", 0.3, 6, min_lr=1e-8)
        cfg.EPOCHS = kwargs["epochs"]
        cfg.SAVE_MODEL = save_model
        trainer = Trainer(cfg)
        trainer.train()

        score = trainer.epoch_metrics["valid_total-loss"]
        #trial.set_user_attr("accuracy", trainer.epoch_metrics["valid_accuracy"])
        trial.set_user_attr("flops", flops(cfg.MODELS[0]))
        return score

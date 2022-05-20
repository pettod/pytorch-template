from sre_constants import IN
from cv2 import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import os
import time
from tqdm import tqdm
from glob import glob

from config import CONFIG
from src.dataset import ImageDataset
from src.trainer import Trainer
from src.utils.utils import saveTensorImage


INPUT_PATHS = sorted(glob("*.png"))
MODEL_PATH = "saved_models/2022-01-28_180347/model_0.pt"
SPARSITY = 0.3
PRUNING_ITERATIONS = 50


def test(model):
    data_transforms = ToTensor()
    dataset = ImageDataset(INPUT_PATHS, None, data_transforms, CONFIG.INPUT_NORMALIZE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    save_directory = os.path.join("predictions", time.strftime("%Y-%m-%d_%H%M%S"))
    with torch.no_grad():
        for i, (x, name) in enumerate(tqdm(dataloader)):
            x = x.cuda()
            x_padding = x.shape[-1] % 2
            y_padding = x.shape[-2] % 2
            if x_padding:
                x = F.pad(x, (1,0,0,0))
            if y_padding:
                x = F.pad(x, (0,0,1,0))
            output = model(x)
            if x_padding:
                output = output[:, :, 1:, :]
            if y_padding:
                output = output[:, :, :, 1:]
            saveTensorImage(
                output,
                os.path.join(save_directory, os.path.basename(name[0])))


def main():
    model = CONFIG.MODELS[0].cuda()
    modules = model.named_modules()
    pruned_parameters = []
    for m in modules:
        if isinstance(m, torch.nn.Conv2d):
            pruned_parameters.append((m, "weight"))
        elif isinstance(m[1], torch.nn.Conv2d):
            pruned_parameters.append((m[1], "weight"))
            # Initialize weight_orig and weight_mask
            # Needed to load pruned model, uncomment loading pruned model
            #prune.identity(m[1], "weight")
    model.load_state_dict(torch.load(MODEL_PATH))
    #model.eval()

    original_parameters = sum(p.numel() for p in model.parameters())
    print("{:,} original parameters".format(original_parameters))

    test(model)
    for i in range(PRUNING_ITERATIONS):

        # Prune parameters
        prune.global_unstructured(
            pruned_parameters,
            pruning_method=prune.L1Unstructured,
            amount=SPARSITY,
        )

        # Remove reparameterization
        # Remove weight_orig because otherwise it's been used to count the
        # number of parameters
        for p in pruned_parameters:
            prune.remove(p[0], p[1])

        # Count parameters
        non_zero_parameters = sum(p.nonzero().size(0) for p in model.parameters())
        print("{:,} parameters after pruning".format(non_zero_parameters))

        # Reapply weight mask for all the zero weights
        prune.global_unstructured(
            pruned_parameters,
            pruning_method=prune.L1Unstructured,
            amount=1 - non_zero_parameters / original_parameters,
        )
        trainer = Trainer(CONFIG.TRAIN_DATASET, CONFIG.VALID_DATASET)
        trainer.train()
        test(model)


main()

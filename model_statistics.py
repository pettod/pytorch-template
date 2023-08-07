import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import itertools
from glob import glob
from importlib import import_module

# Project files
from src.utils.utils import loadModel


MODEL_PATH = sorted(glob("saved_models/*"))[-1]


def loadModels(model_path):
    config = import_module(os.path.join(
        model_path, "codes.config").replace("/", ".")).CONFIG
    model = config.MODELS[0].to(torch.device("cpu"))
    loadModel(model, model_path=model_path)
    return model


def flattenLists(list_of_lists):
    while True:
        try:
            list_of_lists = list(itertools.chain(*list_of_lists))
        except TypeError:
            break
    return list_of_lists


def plotModelStatistics(model, file_name, number_of_bins=200):
    weights = []
    biases = []
    for name, param in model.named_parameters():
        if "weight" in name:
            weights += flattenLists(param.tolist())
        elif "bias" in name:
            biases += flattenLists(param.tolist())

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.title("Weights ({:,.0f})".format(len(weights)))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.hist(weights, number_of_bins)
    plt.subplot(1,2,2)
    plt.title("Biases ({:,.0f})".format(len(biases)))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.hist(biases, number_of_bins)
    plt.tight_layout()
    plt.subplots_adjust(hspace=1.0, wspace=0.3)
    plt.savefig(file_name)


if __name__ == "__main__":
    model = loadModels(MODEL_PATH)
    plotModelStatistics(model, "model_statistics.png")

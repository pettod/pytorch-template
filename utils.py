import glob
from inspect import getmembers, isfunction
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

# Project files
import metrics


def getMetrics():
    metrics_name_and_function_pointers = [
        metric for metric in getmembers(metrics) if isfunction(metric[1])]
    return metrics_name_and_function_pointers


def computeMetrics(y_pred, y_true):
    metric_functions = getMetrics()
    metric_scores = {}
    for metric_name, metric_function_pointer in metric_functions:
        metric_scores[metric_name] = metric_function_pointer(y_pred, y_true)
    return metric_scores


def getEmptyEpochMetrics():
    metric_functions = getMetrics()
    epoch_metrics = {}
    epoch_metrics["train_loss"] = 0
    epoch_metrics["valid_loss"] = 0
    for metric_name, _ in metric_functions:
        epoch_metrics[f"train_{metric_name}"] = 0
        epoch_metrics[f"valid_{metric_name}"] = 0
    return epoch_metrics


def updateEpochMetrics(
        y_pred, y_true, epoch_iteration_index, epoch_metrics, mode):
    metric_scores = computeMetrics(y_pred, y_true)
    for key, value in metric_scores.items():
        if type(value) == torch.Tensor:
            value = value.item()

        epoch_metrics[f"{mode}_{key}"] += ((
            value - epoch_metrics[f"{mode}_{key}"]) /
            (epoch_iteration_index + 1))


def getProgressbarText(epoch_metrics, mode):
    text = f" {mode}:"
    mode = mode.lower()
    for key, value in epoch_metrics.items():
        if mode not in key:
            continue
        text += " {}: {:2.4f}.".format(key.replace(f"{mode}_", ""), value)
    return text


def plotLearningCurve(log_file_path=None, model_root="saved_models/pytorch"):
    # Read CSV log file
    if log_file_path is None:
        log_file_path = sorted(glob.glob(os.path.join(
            model_root, *['*', "*.csv"])))[-1]
    log_file = pd.read_csv(log_file_path)

    # Read data into dictionary
    log_data = {}
    for column in log_file:
        if column == "epoch":
            log_data[column] = np.array(log_file[column].values, dtype=np.str)
        elif column == "learning_rate":
            continue
        else:
            log_data[column] = np.array(log_file[column].values)

    # Remove extra printings of same epoch
    epoch_string_data = []
    previous_epoch = -1
    for epoch in reversed(log_data["epoch"]):
        if epoch != previous_epoch:
            epoch_string_data.append(epoch)
        else:
            epoch_string_data.append('')
        previous_epoch = epoch
    epoch_string_data = epoch_string_data[::-1]
    number_of_rows = len(epoch_string_data)
    log_data.pop("epoch", None)

    # Define train and validation subplots
    figure_dict = {}
    for key in log_data.keys():
        metric = key.split('_')[-1]
        if metric not in figure_dict:
            figure_dict[metric] = len(figure_dict.keys()) + 1
    number_of_subplots = len(figure_dict.keys())

    # Plot learning curves
    plt.figure(figsize=(15, 7))
    import warnings
    warnings.filterwarnings("ignore")
    for i, key in enumerate(log_data.keys()):
        metric = key.split('_')[-1]
        plt.subplot(1, number_of_subplots, figure_dict[metric])
        plt.plot(range(number_of_rows), log_data[key], label=key)
        plt.xticks(range(number_of_rows), epoch_string_data)
        plt.xlabel("Epoch")
        plt.title(metric.upper())
        plt.legend()
    plt.tight_layout()
    plt.savefig("{}.{}".format(log_file_path.split('.')[0], "png"))

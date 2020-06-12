import metrics
from inspect import getmembers, isfunction
import torch


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
import torch


def psnr(y_true, y_pred, max_pixel_value=2.0):
    mse = torch.mean(torch.square(y_true - y_pred))
    if mse == 0:
        return 100
    return 10 * torch.log10(max_pixel_value**2 / mse)

from piq import psnr as psnr_
from piq import ssim as ssim_


def psnr(y_true, y_pred):
    return psnr_(y_true, y_pred)


def ssim(y_true, y_pred):
    return ssim_(y_true, y_pred)

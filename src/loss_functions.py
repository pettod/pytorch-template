import torch
import torch.nn.functional as F


def sobelLoss(y_pred, y_true):
    b, c, h, w = y_true.shape
    sobel_kernel_x = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ]).repeat(b, c, 1, 1)
    sobel_kernel_y = sobel_kernel_x.transpose(2, 3)
    if y_true.is_cuda:
        sobel_kernel_x = sobel_kernel_x.cuda(y_true.get_device())
        sobel_kernel_y = sobel_kernel_y.cuda(y_true.get_device())
    gradients_true_x = F.conv2d(y_true, sobel_kernel_x)
    gradients_true_y = F.conv2d(y_true, sobel_kernel_y)
    gradients_pred_x = F.conv2d(y_pred, sobel_kernel_x)
    gradients_pred_y = F.conv2d(y_pred, sobel_kernel_y)
    gradients_error_x = torch.abs(gradients_true_x - gradients_pred_x)
    gradients_error_y = torch.abs(gradients_true_y - gradients_pred_y)
    return torch.mean(gradients_error_x + gradients_error_y)


def l1(y_pred, y_true):
    return torch.mean(torch.abs(y_true - y_pred))

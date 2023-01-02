import numpy as np
from skimage.util.shape import view_as_windows
import torch.nn.functional as F


# from https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
def conv2d_grad(x, w, b, k, dy):
    dw = F.conv2d(x, weight=dy)
    padded_dy = F.pad(dy, [k - 1] * 4, mode="constant", value=0)
    wt = w.flip([-2, -1])
    dx = F.conv2d(padded_dy, weight=wt)
    # TODO: mmmm unused b is suspicious, I think it works just because top grads are
    # all ones. -> CHECK
    db = dy.sum(axis=(0, 2, 3))
    return dx, dw, db


def conv2d(x, kernel):
    I, H, W = x.shape
    Ik, k1, k2, O = kernel.shape
    assert I == Ik, f"Error: input channels {I} != input weights {Ik}"
    # (1, 28, 28, I, k1, k2) Note: what is the leading 1?
    windows = view_as_windows(x, [I, k1, k2])
    # (28, 28, I, k1 * k2)
    windows = windows.reshape(H - k1 + 1, W - k2 + 1, I, k1 * k2)
    w = kernel.reshape(I, k1 * k2, O)
    # (28, 28, O)
    out = np.tensordot(windows, w)
    return out

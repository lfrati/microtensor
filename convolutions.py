# from https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509

import torch
import torch.nn.functional as F
from rich import print as pprint

# import torch.nn as nn
# conv1d = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=2)
# conv1d.weight.shape -> [out, in, ker]
# conv1d.bias.shape -> [out]

CNT = 32

H, W = 16, 16
BS, I, O = 1, 1, 1
k = 3
x = torch.rand(BS, I, H, W, requires_grad=True)
w = torch.rand(O, I, k, k, requires_grad=True)
b = torch.rand(O, requires_grad=True)
out = F.conv2d(input=x, weight=w, bias=b)
out.retain_grad()
out.sum().backward()

print("inp[BS, I, H, W]", list(x.shape))
print("out[BS, O, ....]", list(out.shape))
# print()
# print(w.grad)

dy = out.grad.detach().clone()

# conv2d expects input [BS, I, H , W ]
# conv2d expects weigh [ O, I, kw, kw]
print("                   [BS, I, H, W]")
print(f" {x.shape=}")
print("                   [ O, I, k, k]")
print(f"{dy.shape=}")
print(f"w.grad   {w.grad.shape}")

dw = F.conv2d(x, weight=dy)
# dw = dw.transpose(1,0)
assert dw.shape == w.shape, f"dw:{list(dw.shape)} != w:{list(w.shape)}"
assert torch.allclose(dw, w.grad)
pprint(f"dw MATCH : [green]OK")

padded_dy = F.pad(dy, [k - 1] * 4, mode="constant", value=0)
wt = w.flip([-2, -1])
dx = F.conv2d(padded_dy, weight=wt)
assert dx.shape == x.shape, f"dx:{list(dx.shape)} != x:{list(x.shape)}"
assert torch.allclose(dx, x.grad), f"{dx=} != {x.grad=}"
pprint(f"dx MATCH : [green]OK")

db = dy.sum(axis=(0, 2, 3))
assert db.shape == b.shape, f"db:{list(db.shape)} != b:{list(b.shape)}"
assert torch.allclose(db, b.grad), f"{db=} != {b.grad=}"
pprint(f"db MATCH : [green]OK")

#%%

import numpy as np
from time import monotonic
from skimage.util.shape import view_as_windows


H, W, I, O = 16, 16, 3, 8
k1 = k2 = 5
x = np.random.rand(I, H, W)
kernel = np.random.rand(I, k1, k2, O)


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


st = monotonic()
for _ in range(CNT):
    out = conv2d(x, kernel)
et = monotonic()
print(f"Numpy: {(et - st)/CNT:.9f}")
# print("out", out, out.shape)


def compare(x, kernel, np_out):
    # add batch
    x = torch.from_numpy(x).unsqueeze(0)
    kernel = torch.from_numpy(kernel.transpose(3, 0, 1, 2))
    st = monotonic()
    for _ in range(CNT):
        torch_out = F.conv2d(input=x, weight=kernel)
    et = monotonic()
    # remove batch
    torch_out = torch_out.squeeze(0)
    print(f"Torch: {(et - st)/CNT:.9f}")
    #  mine is H, W, C (because of the tensordot)
    # torch is C, H, W
    np_out = np_out.transpose(2, 0, 1)
    assert np.allclose(torch_out.numpy(), np_out)
    pprint("CONV MATCH: [green]OK")


compare(x, kernel, out)

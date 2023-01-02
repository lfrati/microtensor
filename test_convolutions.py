import os
from time import monotonic

import numpy as np
from rich import print as pprint
import torch
import torch.nn.functional as F

from convolutions import conv2d_grad, conv2d

CNT = int(os.getenv("CNT", "32"))


def test_conv_grads():

    H, W = 16, 16
    BS, I, O = 1, 1, 1
    k = 3
    x = torch.rand(BS, I, H, W, requires_grad=True)
    w = torch.rand(O, I, k, k, requires_grad=True)
    b = torch.rand(O, requires_grad=True)
    out = F.conv2d(input=x, weight=w, bias=b)
    out.retain_grad()
    out.sum().backward()

    dy = out.grad.detach().clone()

    # conv2d expects input [BS, I, H , W ]
    # conv2d expects weigh [ O, I, kw, kw]
    print("                   [BS, I, H, W]")
    print(f" {x.shape=}")
    print("                   [ O, I, k, k]")
    print(f"{dy.shape=}")
    print(f"w.grad   {w.grad.shape}")

    dx, dw, db = conv2d_grad(x, w, b, k, dy)

    # dw = dw.transpose(1,0)
    assert dw.shape == w.shape, f"dw:{list(dw.shape)} != w:{list(w.shape)}"
    assert torch.allclose(dw, w.grad)
    pprint(f"dw MATCH : [green]OK")

    assert dx.shape == x.shape, f"dx:{list(dx.shape)} != x:{list(x.shape)}"
    assert torch.allclose(dx, x.grad), f"{dx=} != {x.grad=}"
    pprint(f"dx MATCH : [green]OK")

    assert db.shape == b.shape, f"db:{list(db.shape)} != b:{list(b.shape)}"
    assert torch.allclose(db, b.grad), f"{db=} != {b.grad=}"
    pprint(f"db MATCH : [green]OK")


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


def test_conv_numpy():

    H, W, I, O = 16, 16, 3, 8
    k1 = k2 = 5
    x = np.random.rand(I, H, W)
    kernel = np.random.rand(I, k1, k2, O)

    st = monotonic()
    for _ in range(CNT):
        out = conv2d(x, kernel)
    et = monotonic()
    print(f"Numpy: {(et - st)/CNT:.9f}")
    # print("out", out, out.shape)

    compare(x, kernel, out)

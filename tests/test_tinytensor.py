import numpy as np
from rich import print as pprint
from tinytensor import Linear, Tensor
import torch
import torch.nn.functional as F


def compare(v_torch, v, name):
    grad = v_torch.grad.numpy()
    assert np.allclose(grad, v.grad.data), f"{name.upper()} DO NOT MATCH"
    pprint(f"{name.upper():>8} MATCH : [green]OK")


def test_linear():

    x = Tensor.randn(3, 8)
    fc = Linear(in_features=8, out_features=4)
    a3 = fc(x).relu()
    a4 = a3.sum()
    a4.backward()

    x_torch = torch.from_numpy(x.data.copy())
    w_torch = torch.from_numpy(fc.w.data.copy())
    b_torch = torch.from_numpy(fc.b.data.copy())
    x_torch.requires_grad = True
    w_torch.requires_grad = True
    b_torch.requires_grad = True
    a3_torch = F.linear(input=x_torch, weight=w_torch.T, bias=b_torch).relu()
    a4_torch = a3_torch.sum()

    a4_torch.backward(retain_graph=True)

    compare(x_torch, x, "xs")
    compare(w_torch, fc.w, "weights")
    compare(b_torch, fc.b, "biases")

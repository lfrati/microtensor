from microtensor import Tensor
import numpy as np
from rich import print as pprint
import torch


def compare(v_torch, v, name):
    grad = v_torch.grad.numpy()
    assert np.allclose(grad, v.grad), f"{name.upper()} DO NOT MATCH"
    pprint(f"{name.upper():>8} MATCH : [green]OK")


def test_linear():
    x = Tensor.randn(3, 8)
    w = Tensor.uniform(8, 4)
    b = Tensor.uniform(1, 4)

    a1 = x @ w
    bb = b.broadcast(a1)
    a2 = a1 + bb
    a3 = a2.sum()

    print(x)
    print(w)
    print(b)
    print(a1)
    print(a2)
    print(a3)

    a3.backward()

    pprint("[red]a3\n", a3.grad)
    pprint("[red]a2\n", a2.grad)
    pprint("[red]a1\n", a1.grad)
    pprint("[red]x\n", x.grad)
    pprint("[red]w\n", w.grad)
    pprint("[red]b.d\n", b.data)
    pprint("[red]b.g\n", b.grad)
    print()

    x_torch = torch.from_numpy(x.data.copy())
    w_torch = torch.from_numpy(w.data.copy())
    b_torch = torch.from_numpy(b.data.copy())
    x_torch.requires_grad = True
    w_torch.requires_grad = True
    b_torch.requires_grad = True
    a1_torch = x_torch.matmul(w_torch)
    a2_torch = a1_torch + b_torch
    a3_torch = a2_torch.sum()

    a3_torch.backward(retain_graph=True)

    compare(x_torch, x, "xs")
    compare(w_torch, w, "weights")
    compare(b_torch, b, "biases")

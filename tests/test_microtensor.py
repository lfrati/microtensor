import numpy as np
import pytest
from rich import print as pprint
import torch

from microtensor import Tensor


def compare(v_torch, v, name):
    grad = v_torch.grad.numpy()
    assert np.allclose(grad, v.grad), f"{name.upper()} DO NOT MATCH"
    pprint(f"{name.upper():>8} MATCH : [green]OK")


@pytest.fixture
def forward():
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

    state = {}
    state["x"] = x
    state["w"] = w
    state["b"] = b
    state["x_torch"] = x_torch
    state["w_torch"] = w_torch
    state["b_torch"] = b_torch
    state["out"] = a3
    state["out_torch"] = a3_torch

    return state


def test_input(forward):
    compare(forward["x_torch"], forward["x"], "xs")


def test_weight(forward):
    compare(forward["w_torch"], forward["w"], "weights")


def test_bias(forward):
    compare(forward["b_torch"], forward["b"], "biases")

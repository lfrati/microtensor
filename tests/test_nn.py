from nn import Linear, MLP, SGD
from tinytensor import Tensor
import numpy as np
from rich import print as pprint
import torch
import torch.nn.functional as F


def compare(v_torch, v, name):
    grad = v_torch.grad.numpy()
    assert np.allclose(grad, v.grad.data), f"{name.upper()} DO NOT MATCH"
    pprint(f"{name.upper():>8} MATCH : [green]OK")


def test_mlp():

    net = MLP(8, 32, 10)
    optim = SGD(net.parameters(), lr=0.001)
    xs = Tensor.randn(3, 8)

    # print(net)
    #
    # for p in net.parameters():
    #     print(p.shape)
    #
    # print()
    #
    # for name, p in net.named_parameters():
    #     print(name, p.shape)
    #
    # print()
    # print("Forwarding")

    out = net(xs).sum()

    start_val = out.data.copy()

    for _ in range(5):

        out = net(xs).sum()
        out.backward()

        optim.step()
        optim.zero_grad()

    end_val = out.data.copy()

    assert end_val < start_val


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

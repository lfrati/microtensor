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


def test_xor():

    net = MLP(2, 32, 1)
    optim = SGD(net.parameters(), lr=0.01)
    xs = Tensor.from_list([[0, 0], [0, 1], [1, 0], [1, 1]])
    ys = Tensor.from_list([[0], [1], [1], [0]])

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

    def squared_err(x, y):
        return ((x - y) ** Tensor.from_list(2)).sum()

    out = net(xs)
    loss = squared_err(out, ys)

    start_loss = loss.data.copy()

    for _ in range(200):

        out = net(xs)
        loss = squared_err(out, ys)
        print(loss.data)
        loss.backward()

        optim.step()
        optim.zero_grad()

    end_loss = loss.data.copy()
    final_pred = np.where(out.data > 0.5, 1.0, 0.0)

    print(f"Loss decreade from {start_loss} to {end_loss}")
    print("Expected", ys.data.T)
    print("     Got", final_pred.T)

    assert end_loss < start_loss
    assert np.allclose(final_pred, ys.data)


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

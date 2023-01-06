import numpy as np
from tinytensor import Tensor


def test_add():
    a = Tensor.randn(3, 8)
    b = Tensor.randn(3, 8)

    c = a + b

    assert np.allclose(a.data + b.data, c.data)


def test_add():
    a = Tensor.randn(3, 8)
    b = Tensor.randn(3, 8)

    c = a * b

    assert np.allclose(a.data * b.data, c.data)


def test_matmul():
    a = Tensor.randn(3, 8)
    b = Tensor.randn(8, 3)

    c = a @ b

    assert np.allclose(a.data @ b.data, c.data)


def test_sum():
    a = Tensor.randn(3, 8)

    c = a.sum()

    assert np.allclose(a.data.sum(keepdims=True), c.data)


def test_broadcast():

    a = Tensor.randn(4, 8)
    b = Tensor.randn(1, 8)

    c = a + b.broadcast(a)

    assert np.allclose(a.data + b.data, c.data)


def test_relu():

    a = Tensor.randn(8, 512)

    c = a.relu()

    assert np.allclose(np.maximum(a.data, 0), c.data)

import numpy as np
from rich import print as pprint

"""
Extension of micrograd to use tensors instead of single values.
"""


def broadcast_dims(a, b):
    err_msg = f"Broadcasting missing dimensions is not supported: {len(a.shape)=} != {len(b.shape)=}"
    assert len(a.shape) == len(b.shape), err_msg
    shapes = list(zip(a.shape, b.shape))
    dims = []
    for i, (d1, d2) in enumerate(shapes):
        if d1 == 1 and d2 > 1:
            dims.append(i)
    return dims


class Tensor:
    def __init__(self, data: np.ndarray, _parents=(), _op="."):
        self.data = data
        self.shape = data.shape
        self._parents = _parents
        self._op = _op
        self.grad = np.zeros_like(data)
        self._backward = lambda: None

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape=shape, dtype=np.float32), **kwargs)

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape=shape, dtype=np.float32), **kwargs)

    @classmethod
    def uniform(cls, *shape, **kwargs):
        data = np.random.default_rng().random(size=shape, dtype=np.float32) * 2 - 1
        return cls(data, **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        data = np.random.default_rng().standard_normal(size=shape, dtype=np.float32)
        return cls(data, **kwargs)

    def broadcast(self, other):
        dims = broadcast_dims(self, other)
        if len(dims) > 0:
            broadcasted = np.broadcast_to(self.data, other.shape)
            out = Tensor(broadcasted, _parents=(self,), _op="B")

            def backward():
                self.grad += out.grad.sum(axis=tuple(dims))

            out._backward = backward
            return out

        return self

    def __matmul__(self, other):
        data = self.data @ other.data
        out = Tensor(data, _parents=(self, other), _op="@")

        def backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = backward
        return out

    def __add__(self, other):
        data = self.data + other.data
        out = Tensor(data, _parents=(self, other), _op="+")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __mul__(self, other):
        data = self.data * other.data
        out = Tensor(data, _parents=(self, other), _op="*")

        def backward():
            self.grad += out.grad * other.data
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def sum(self):
        v = self.data.sum()
        out = Tensor(v, _parents=(self,))

        def backward():
            self.grad += np.ones_like(self.grad) * out.grad

        out._backward = backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def __repr__(self):
        return f"Tensor(data:{self.shape}, grad:{'None' if self.grad.sum() == 0 else 'Yes'}, op:{self._op})"

    def toposort(self):
        def _toposort(node, visited, nodes):
            visited.add(node)
            for i in node._parents:
                if i not in visited:
                    _toposort(i, visited, nodes)
            nodes.append(node)
            return reversed(nodes)

        return _toposort(self, set(), [])

    def backward(self):
        assert self.shape == ()

        self.grad = np.ones(shape=self.shape)

        for t0 in self.toposort():
            t0._backward()

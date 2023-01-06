import functools
from typing import Optional

import numpy as np
from rich import print as pprint

"""
TinyNotes:

While micrograd defines __add__, __mul__, ...
TinyGrad uses a much more flexible but much intricate approach:

Operations are first defined in tinygrad/tensor.py:

    class Function:
      def __init__(self, device:str, *tensors:Tensor):
        self.device, self.parents = device, tensors
        self.needs_input_grad = ...
        self.requires_grad = ...
        self.saved_tensors : List[Tensor] = []
      def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
      def backward(self, *args, **kwargs): raise NotImplementedError(f"backward not implemented for {type(self)}")
      def save_for_backward(self, *x): self.saved_tensors.extend(x)

      @classmethod
      def apply(cls, *x:Tensor, **kwargs):
        ...

But actual functions are in tinygrad/mlops.py

    class Add(Function):
        ...

    class Mul(Function):
      def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.binary_op(BinaryOps.MUL, y)

      def backward(self, grad_output):
        return self.saved_tensors[1].binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
               self.saved_tensors[0].binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None


When creating tensor.py:
1. Get functions names
2. lowercase them
3. register them on tensors starting with "_" (e.g. Add -> _add, Mul -> _mul)

So in tinygrad/tensor.py:

    def register(name:str, fxn:Function):
      setattr(Tensor, "_"+name if hasattr(Tensor, name) else name, functools.partialmethod(fxn.apply))
    for name, cls in inspect.getmembers(importlib.import_module('tinygrad.mlops'), inspect.isclass):
      if name[0] != "_" and name != "Function" and not name.endswith("Ops"):
        register(name.lower(), cls)

Then define add to call _add, mul to call _mul
In tinygrad/tensor.py

    def add(self, x): return Tensor.broadcasted(Tensor._add, self, x)
    def mul(self, x): return Tensor.broadcasted(Tensor._mul, self, x)

Finally register the __add__, __mul__ to call add, mul, ...

    # register the operators
    def register_op(name, fxn):
      setattr(Tensor, f"__{name}__", fxn)
      setattr(Tensor, f"__i{name}__", lambda self,x: self.assign(fxn(self,x)))
      setattr(Tensor, f"__r{name}__", lambda self,x: fxn(x,self))
    for name in ['add', 'sub', 'mul', 'pow', 'matmul', 'truediv']:
      register_op(name, getattr(Tensor, name if name != 'truediv' else 'div'))

Ok, but what do these function actually do?
Notice for example the Add.forward -> return x.binary_op(BinaryOps.ADD, y)

What is this? The "actual" implementation is specified in the different backends:

For the CPU -> llops/ops_cpu
    BinaryOps.ADD: operator.add

For the GPU -> llops/ops_gpu
    BinaryOps.ADD: "(A+B)" # something to do with CL, haven't check the implementation

"""


def broadcast_dims(a, b):
    err_msg = f"Broadcasting missing dimensions is not supported: {len(a.shape)=} != {len(b.shape)=}"
    assert len(a.shape) == len(b.shape), err_msg
    shapes = list(zip(a.shape, b.shape))
    dims = []
    for i, (d1, d2) in enumerate(shapes):
        if d1 == 1 and d2 > 1:
            dims.append(i)
    return tuple(dims)


class Parameter:
    def parameters(self):
        return [self]

    def named_parameters(self):
        return [("", self)]


class Tensor(Parameter):
    def __init__(self, data: np.ndarray):
        self.data = data
        self.shape = data.shape
        self.grad = None
        self._ctx: Optional[Function] = None

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape=shape, dtype=np.float32), **kwargs)

    @classmethod
    def full(cls, shape, fill_value, **kwargs):
        return cls(
            np.full(shape=shape, fill_value=fill_value, dtype=np.float32), **kwargs
        )

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

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __repr__(self):
        return f"Tensor({self.data} shape:{self.shape}, grad:{self.grad})"

    def toposort(self):
        def _toposort(node, visited, nodes):
            visited.add(node)
            if node._ctx:
                for i in node._ctx.parents:
                    if i not in visited:
                        _toposort(i, visited, nodes)
            nodes.append(node)
            return reversed(nodes)

        return _toposort(self, set(), [])

    def backward(self, retain=False):
        assert (
            self.shape == ()
        ), f"implicit differentiation requires scalar input: got {self.shape=}"

        self.grad = Tensor.ones(*self.shape)

        for node in self.toposort():
            assert node.grad is not None
            if not node._ctx:  # leaves have no context
                continue

            # unwrap/wrap tensor data to decouple operations from storage
            grads = node._ctx.backward(node.grad.data)
            if len(node._ctx.parents) == 1:
                grads = [Tensor(grads)]
            else:
                grads = [Tensor(g) if g is not None else None for g in grads]
            for parent, grad in zip(node._ctx.parents, grads):
                if grad is not None:
                    assert (
                        grad.shape == parent.shape
                    ), f"grad shape must match tensor shape: {grad.shape!r} != {parent.shape!r}"
                    parent.grad = grad if parent.grad is None else (parent.grad + grad)
            if not retain:
                del node._ctx


# An instantiation of the Function is the Context
class Function:
    def __init__(self, *parents: Tensor):
        self.parents = parents
        self.saved_tensors: list[np.ndarray] = []

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @classmethod
    def apply(cls, *x: Tensor, **kwargs):
        ctx = cls(*x)
        ret = Tensor(ctx.forward(*[t.data for t in x], **kwargs))
        ret._ctx = ctx  # used by autograd engine
        return ret


class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, grad_output):
        return (grad_output, grad_output)


class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x * y

    def backward(self, grad_output):
        return (
            grad_output * self.saved_tensors[1],
            self.saved_tensors[0] * grad_output,
        )


class Matmul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, grad_output):
        return (
            grad_output @ self.saved_tensors[1].T,
            self.saved_tensors[0].T @ grad_output,
        )


class Sum(Function):
    def forward(self, x):
        self.input_shape = x.shape
        return x.sum()

    def backward(self, grad_output):
        return np.ones(shape=self.input_shape) * grad_output


class Broadcast(Function):
    def forward(self, x, y):
        self.input_shape = x.shape
        self.dims_to_reduce = broadcast_dims(x, y)
        assert (
            len(self.dims_to_reduce) > 0
        ), f"Dumb broadcast detected: {y.shape=} -> {x.shape=}"
        return np.broadcast_to(x, y.shape)

    def backward(self, grad_output):
        out = grad_output.sum(axis=self.dims_to_reduce, keepdims=True)
        return out, None


class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.maximum(x, 0)

    def backward(self, grad_output):
        return (self.saved_tensors[0] > 0) * grad_output


def register(name: str, fxn: Function):
    setattr(Tensor, name, functools.partialmethod(fxn.apply))


register("__add__", Add)
register("__mul__", Mul)
register("__matmul__", Matmul)
register("sum", Sum)
register("broadcast", Broadcast)
register("relu", ReLU)

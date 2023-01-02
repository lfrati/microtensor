import numpy as np
import torch
from rich import print as pprint

"""
Extension of micrograd to use tensors instead of single values.
"""

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

    import numpy as np

    x = np.random.rand(3, 5) * 2 - 1
    print(x)
    print()

    def __repr__(self):
        return f"Tensor(data:{self.shape}, grad:{'None' if self.grad.sum() == 0 else 'Yes'}, op:{self._op})"

    def deepwalk(self):
        def _deepwalk(node, visited, nodes):
            visited.add(node)
            for i in node._parents:
                if i not in visited:
                    _deepwalk(i, visited, nodes)
            nodes.append(node)
            return reversed(nodes)

        return _deepwalk(self, set(), [])

    def backward(self):
        assert self.shape == ()

        self.grad = np.ones(shape=self.shape)

        for t0 in self.deepwalk():
            t0._backward()


#%%

if __name__ == "__main__":

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

    #%%

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

    def compare(v_torch, v, name):
        grad = v_torch.grad.numpy()
        assert np.allclose(grad, v.grad), f"{name.upper()} DO NOT MATCH"
        pprint(f"{name.upper():>8} MATCH : [green]OK")

    compare(x_torch, x, "xs")
    compare(w_torch, w, "weights")
    compare(b_torch, b, "biases")

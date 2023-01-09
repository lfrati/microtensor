<h1 align="center">
   µTens⨂r
   <p></p>
</h1>

([micrograd](https://github.com/karpathy/micrograd) + [tinygrad](https://github.com/geohot/tinygrad))/2

Micrograd and Tinygrad use different styles:
- [microtensor](./microtensor.py) implements the micrograd style + tensors
```python
def __mul__(self, other):
        data = self.data * other.data
        out = Tensor(data, _parents=(self, other), _op="*")

        def backward():
            self.grad += out.grad * other.data
            other.grad += self.data * out.grad

        out._backward = backward
        return out
```
- [tinytensor](./tinytensor.py) implements a simplified version of the tinygrad/pytorch style:
```python
class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x * y

    def backward(self, grad_output):
        return (
            grad_output * self.saved_tensors[1],
            self.saved_tensors[0] * grad_output,
        )
        
setattr(Tensor, "__mul__", functools.partialmethod(Mul.apply))
```
While tinygrad is beautifully flexible and supports multiple backends we assume numpy/cpu backend for simplicity.

The [nn.py](./nn.py) module is inspired by torch's [nn.Module](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module).

## Convolutions
Convolutions are pretty cool because turns out the backward pass can be implemented as convolutions too (kinda like matmuls are transposed matmuls on the backward pass).

Image from [neodelphis](https://neodelphis.github.io/convnet/maths/python/english/2019/07/10/convnet-bp-en.html):

<img width="400" alt="image" src="https://user-images.githubusercontent.com/3115640/211423054-dafb7b61-a6b4-4683-9f0d-68e89d8ee970.png">

[convolutions.py](./convolutions.py) implements a strided-numpy version of convolutions and gradients FOR THE SINGLE CHANNEL CASE.
- [ ] Make convolutions work for multi channels in/out.

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
While tinygrad is beautifully general to support multiple backends we assume numpy/cpu backend for simplicity.

from tinytensor import Parameter, Tensor


class Linear(Parameter):
    def __init__(self, in_features, out_features, bias_value=0.01):
        self.in_features = in_features
        self.out_features = out_features
        self.w = Tensor.uniform(in_features, out_features)
        self.b = Tensor.full(shape=(1, out_features), fill_value=bias_value)

    def __call__(self, x):
        x = x @ self.w
        x = x + self.b.broadcast(x)
        return x

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features})"

    def parameters(self):
        return [self.w, self.b]

    def named_parameters(self):
        return [("w", self.w), ("b", self.b)]


class Module:
    _parameters = []

    def __init__(self):
        super().__setattr__("_parameters", [])

    def __setattr__(self, name: str, value) -> None:
        if issubclass(value.__class__, Parameter):
            params: list = self.__dict__.get("_parameters")
            params.append((name, value))
        super().__setattr__(name, value)

    def parameters(self):
        return [p for _, layer in self._parameters for p in layer.parameters()]

    def named_parameters(self):
        return [
            (".".join([name1, name2]) if name2 != "" else name1, p)
            for name1, layer in self._parameters
            for name2, p in layer.named_parameters()
        ]

    def __repr__(self):
        params = "\n".join(
            ["  " + name + " : " + str(p) for (name, p) in self._parameters]
        )
        return f"{self.__class__.__name__}(\n{params}\n)"


class MLP(Module):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.fc1 = Linear(in_features=in_features, out_features=hid_features)
        self.fc2 = Linear(in_features=hid_features, out_features=out_features)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x


class SGD:
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad.data

    def zero_grad(self):
        for p in self.params:
            p.grad = None

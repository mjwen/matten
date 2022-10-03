import torch
from e3nn.io import CartesianTensor


class CartesianTensorWrapper:
    """
    A wrapper of CartesianTensor that keeps a copy of reduced tensor product to
    avoid memory leak.
    """

    def __init__(self, formula):
        self.converter = CartesianTensor(formula=formula)
        self.rtp = self.converter.reduced_tensor_products()

    def from_cartesian(self, data):
        return self.converter.from_cartesian(data, self.rtp.to(data.device))

    def to_cartesian(self, data):
        return self.converter.to_cartesian(data, self.rtp.to(data.device))


class ToCartesian(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.ct = CartesianTensorWrapper(formula)

    def forward(self, data):
        return self.ct.to_cartesian(data)

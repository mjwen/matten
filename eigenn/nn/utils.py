from typing import Callable, Optional

import torch
import torch.nn.functional as fn
from e3nn.io import CartesianTensor as E3NNCartesianTensor
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, TensorProduct
from nequip.nn.nonlinearities import ShiftedSoftPlus
from torch import Tensor

ACTIVATION = {
    # for even irreps
    "e": {
        "ssp": ShiftedSoftPlus,
        "silu": fn.silu,
    },
    # for odd irreps
    "o": {
        "abs": torch.abs,
        "tanh": torch.tanh,
    },
}


# TODO this has been PRed to e3nn, use that one
class CartesianTensor(E3NNCartesianTensor):
    def to_cartesian(self, irreps_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert an irreps tensor to a Cartesian tensor.

        This is the inverse operation of `from_cartesian`.

        Args:
            irreps_tensor: irreps tensor of shape (..., D), and `D` is the dim of the
                irreps data, and ... indicates additional axes can be added, e.g.
                batch dimensions.

        Returns:
            the tensor in Cartesian view
        """

        Q = self.change_of_basis()
        cart_tensor = irreps_tensor @ Q.flatten(-self.num_index)

        shape = list(irreps_tensor.shape[:-1]) + list(Q.shape[1:])
        cart_tensor = cart_tensor.view(shape)

        return cart_tensor


class UVUTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        *,
        internal_and_share_weights: bool = False,
        mlp_input_size: int = None,
        mlp_hidden_size: int = 8,
        mlp_num_hidden_layers: int = 1,
        mlp_activation: Callable = ACTIVATION["e"]["ssp"],
    ):
        """
        UVU tensor product as in NeuqIP.

        Args:
            irreps_in1: irreps of first input, with available keys in `DataKey`
            irreps_in2: input of second input, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`
            internal_and_share_weights: whether to create weights for the tensor
                product, if `True` all `mlp_*` params are ignored and if `False`,
                they should be provided to create an MLP to transform some data to be
                used as the weight of the tensor product.
            mlp_input_size: size of the input data used as the weight for the tensor
                product transformation via an MLP
            mlp_hidden_size: hidden layer size for the MLP
            mlp_num_hidden_layers: number of hidden layers for the radial MLP, excluding
                input and output layers
            mlp_activation: activation function for the MLP.
        """

        super().__init__()

        # uvu instructions for tensor product
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_in1):
            for j, (_, ir_in2) in enumerate(irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # sort irreps_mid so we can simplify them later
        irreps_mid = Irreps(irreps_mid)
        self.irreps_mid, permutation, _ = irreps_mid.sort()

        assert irreps_mid.dim > 0, (
            f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} produces no "
            f"instructions in irreps_out={irreps_out}"
        )

        # sort instructions accordingly
        instructions = [
            (i_1, i_2, permutation[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            irreps_in1,
            irreps_in2,
            self.irreps_mid,
            instructions,
            internal_weights=internal_and_share_weights,
            shared_weights=internal_and_share_weights,
        )

        if not internal_and_share_weights:
            # radial network on scalar edge embedding (e.g. edge distance)
            layer_sizes = (
                [mlp_input_size]
                + mlp_num_hidden_layers * [mlp_hidden_size]
                + [self.tp.weight_numel]
            )
            self.weight_nn = FullyConnectedNet(layer_sizes, act=mlp_activation)
        else:
            self.weight_nn = None

    def forward(
        self, data1: Tensor, data2: Tensor, data_weight: Optional[Tensor] = None
    ) -> Tensor:

        if self.weight_nn is not None:
            assert data_weight is not None, "data for weight not provided"
            weight = self.weight_nn(data_weight)
        else:
            weight = None
        x = self.tp(data1, data2, weight)

        return x

    @property
    def irreps_out(self):
        """
        Output irreps of the layer.

        This is different from the input `irreps_out`, since we we use the UVU tensor
        product with given instructions.
        """
        return self.irreps_mid


if __name__ == "__main__":

    # general 2D tensor
    t = CartesianTensor("ij=ij")

    data = torch.arange(2 * 3 * 3).to(torch.float).view(2, 3, 3)

    ir_t = t.from_cartesian(data)
    cart_t = t.to_cartesian(ir_t)

    print(data)
    print(cart_t)

    assert torch.allclose(data, cart_t)

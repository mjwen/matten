import torch
import torch.nn.functional as fn
from e3nn.io import CartesianTensor as E3NNCartesianTensor
from e3nn.o3 import Irreps
from nequip.nn.nonlinearities import ShiftedSoftPlus

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


def get_uvu_instructions(irreps_in1: Irreps, irreps_in2: Irreps, irreps_out: Irreps):
    """
    Get the instructions for the uvu tensor product.

    This is a helper function to support two-step tensor product:
    1. irreps_in1 \otimes irreps_in2 -> irreps_mid
    2. linear(irreps_mid) -> irreps_out

    This function prepares the instructions and irreps_mid for step 1.

    Args:
        irreps_in: the irreps of the input node features
        irreps_out: the irreps of the output node features

    Returns:
        instructions: sorted instructions
        irreps_mid: sorted irreps (should not simplified) for the mid layer
    """

    # uvu instructions
    irreps_mid = []
    instructions = []
    for i, (mul, ir_in1) in enumerate(irreps_in1):
        for j, (_, ir_in2) in enumerate(irreps_in2):
            for ir_out in ir_in1 * ir_in2:
                if ir_out in irreps_out:
                    k = len(irreps_mid)
                    irreps_mid.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))

    # sort irreps_mid so we can simplify them later in the linear layer
    irreps_mid = Irreps(irreps_mid)
    irreps_mid, permutation, _ = irreps_mid.sort()

    assert irreps_mid.dim > 0, (
        f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} produces no "
        f"instructions in irreps_out={irreps_out}"
    )

    # sort instructions accordingly
    instructions = [
        (i_1, i_2, permutation[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions
    ]

    return instructions, irreps_mid


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


if __name__ == "__main__":

    # general 2D tensor
    t = CartesianTensor("ij=ij")

    data = torch.arange(2 * 3 * 3).to(torch.float).view(2, 3, 3)

    ir_t = t.from_cartesian(data)
    cart_t = t.to_cartesian(ir_t)

    print(data)
    print(cart_t)

    assert torch.allclose(data, cart_t)

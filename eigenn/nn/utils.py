import torch
from e3nn.io import CartesianTensor as E3NNCartesianTensor


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

import torch

from eigenn.nn.uitls import CartesianTensor


def test_cartesian_tensor():
    def assert_one(data):

        # general 2D tensor
        t = CartesianTensor("ij=ij")

        ir_t = t.from_cartesian(data)
        cart_t = t.to_cartesian(ir_t)

        assert torch.allclose(data, cart_t, atol=1e-5)

    data = torch.arange(3 * 3).to(torch.float).view(3, 3)
    assert_one(data)

    # test batched
    data = torch.arange(2 * 4 * 3 * 3).to(torch.float).view(2, 4, 3, 3)
    assert_one(data)

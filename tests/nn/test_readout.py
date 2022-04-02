import torch

from eigenn.data.irreps import DataKey
from eigenn.nn.readout import IrrepsToCartesianTensor, IrrepsToHessian


def test_irreps_to_cartesian():

    i2c = IrrepsToCartesianTensor(
        irreps_in={DataKey.NODE_FEATURES: "0e + 2e"},
        formula="ij=ji",
        out_field="my_out",
    )

    t = torch.arange(9).to(torch.float).view(3, 3)
    y = i2c.ct.from_cartesian(t)

    data = {DataKey.NODE_FEATURES: y}
    out = i2c(data)
    z = out["my_out"]

    assert torch.allclose(z, (t + t.T) / 2, atol=1e-5)


def test_irreps_to_hessian():

    i2h = IrrepsToHessian(
        irreps_in={DataKey.NODE_FEATURES: "0e + 1e + 2e"},
        out_field="my_out",
        symmetrize=False,
    )

    natoms = 2
    t = torch.arange(natoms * 9).to(torch.float).view(natoms, 3, 3)
    y = i2h.ct.from_cartesian(t)
    assert y.shape == (natoms, 9)  # 9 corresponds to '0e+1e+2e'

    layout = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    data = {
        DataKey.NODE_FEATURES: y,
        "hessian_off_diag_layout": layout,
        "ptr": torch.tensor([0, 2]),  # used to get the number of atoms in a config
    }

    out = i2h(data)
    z = out["my_out"]
    assert z.shape == (4, 3, 3)

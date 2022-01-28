from typing import Dict, Optional

import torch
from e3nn.io import CartesianTensor
from e3nn.o3 import FullyConnectedTensorProduct, Irreps

from eigenn.data.irreps import DataKey, ModuleIrreps
from eigenn.nn.utils import tp_path_exists


class IrrepsToCartesianTensor(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        formula: str = "ij=ji",
        field: str = DataKey.NODE_FEATURES,
        out_field: Optional[str] = None,
    ):
        """
        Convert irreps tensor to cartesian tensor.

        Args:
            irreps_in:
            formula: formula to indicate the symmetry of the cartesian tensor,
                e.g. `ij=ij` means a general 2D tensor. See the docs of
                `CartesianTensor` for more.
            field:
            out_field:
        """
        super().__init__()

        self.formula = formula
        self.field = field
        self.out_field = field if out_field is None else out_field

        # NOTE, should not add output to irreps_out, since it is a cartesian tensor,
        # no longer an irreps
        self.init_irreps(irreps_in, required_keys_irreps_in=[field])

        self.ct = CartesianTensor(formula=formula)

        assert self.irreps_in[self.field] == self.ct, (
            f"input irreps of {self.field} is {self.irreps_in[self.field]}, not equal "
            f"to the irreps of the target irreps {self.ct}"
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        value = data[self.field]
        cartesian_tensor = self.ct.to_cartesian(value)
        data[self.out_field] = cartesian_tensor

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  field: {self.field}, out_field: {self.out_field}, formula: {self.formula}\n"
            ")"
        )


class IrrepsToHessian(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        field: str = DataKey.NODE_FEATURES,
        out_field: Optional[str] = None,
    ):
        """
        Convert irreps tensor to 3N by 3N Hessian matrix of a configuration.

        Each 3x3 sub matrix in the rows 3i~3i+3 and columns 3j~3j+3 of the Hessian
        matrix is constructed from the node features of atom i, and atom j (by
        mapping the tensor products of the node features to a general 3x3 matrix).

        Args:
            irreps_in:
            field:
            out_field:
        """
        super().__init__()

        self.field = field
        self.out_field = field if out_field is None else out_field

        # NOTE, should not add output to irreps_out, since it is a cartesian tensor,
        # no longer an irreps
        self.init_irreps(irreps_in, required_keys_irreps_in=[field])

        # for each sub 3x3 matrix of the hessian, it is a general 2D matrix
        self.ct = CartesianTensor(formula="ij=ij")  # i.e. 0e+1e+2e

        # ensure tp between two irreps_in can product self.ct
        for mul, ir in self.ct:
            assert tp_path_exists(
                self.irreps_in[self.field], self.irreps_in[self.field], ir
            ), f"Product between two irreps_in cannot product {ir}"

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in[self.field], self.irreps_in[self.field], self.ct
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        node_feats = data[self.field]

        layout = data["hessian_layout"]
        row_index = layout[:, 0]
        column_index = layout[:, 1]

        # convert node feats to node pair feats, with of shape
        # shape of x: (num_index, 9), where num_index = N1^2 +N2^2 + ..., in which
        # N1, N2... are the number of atoms in the batched molecules
        x = self.tp(node_feats[row_index], node_feats[column_index])

        # convert irreps tensor to cartesian tensor
        cartesian_tensor = self.ct.to_cartesian(x)  # (num_index, 3, 3)

        data[self.out_field] = cartesian_tensor

        return data

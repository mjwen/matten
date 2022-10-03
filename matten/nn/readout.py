from typing import Dict, Optional

import torch
from e3nn.io import CartesianTensor
from e3nn.nn import Extract
from e3nn.o3 import FullTensorProduct, FullyConnectedTensorProduct, Irreps

from matten.data.irreps import DataKey, ModuleIrreps
from matten.dataset.hessian import symmetrize_hessian
from matten.nn.utils import tp_path_exists


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
        self.ct_rtp = self.ct.reduced_tensor_products()

        assert self.irreps_in[self.field] == self.ct, (
            f"input irreps of {self.field} is {self.irreps_in[self.field]}, not equal "
            f"to the irreps of the target irreps {self.ct}"
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        value = data[self.field]
        cartesian_tensor = self.ct.to_cartesian(value, self.ct_rtp.to(value.device))
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
        symmetrize: bool = True,
    ):
        """
        Convert irreps tensor to 3N by 3N Hessian matrix of a configuration.

        Each 3x3 sub matrix in the rows 3i~3i+3 and columns 3j~3j+3 of the Hessian
        matrix is constructed from the node features (0e+1e+2e) of atom i, and atom
        j (by mapping the tensor products of the node features to a general 3x3 matrix).

        Args:
            irreps_in:
            field:
            out_field:
            symmetrize: whether to symmetrize the output by (H + H^T)/2, where T
                denote transpose.
        """
        super().__init__()

        self.field = field
        self.out_field = field if out_field is None else out_field
        self.symmetrize = symmetrize

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

        layout = data["hessian_off_diag_layout"]
        row_index = layout[:, 0]
        column_index = layout[:, 1]

        # convert node feats to node pair feats, with of shape
        # of x: (num_index, 9), where num_index = N1^2 +N2^2 + ..., in which
        # N1, N2... are the number of atoms in the batched molecules
        x = self.tp(node_feats[row_index], node_feats[column_index])

        # convert irreps tensor to cartesian tensor
        cartesian_tensor = self.ct.to_cartesian(x)  # (num_index, 3, 3)

        if self.symmetrize:
            # number of atoms of each graph
            ptr = data["ptr"]
            natoms = [j - i for i, j in zip(ptr[:-1], ptr[1:])]
            cartesian_tensor = symmetrize_hessian(cartesian_tensor, natoms)

        data[self.out_field] = cartesian_tensor

        return data


class IrrepsToIrrepsHessian(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        field: str = DataKey.NODE_FEATURES,
        out_field: Optional[str] = None,
    ):
        """
        Convert irreps tensor on individual atoms to irreps tensor for 3 by 3 Hessian
        blocks.

        For the diagonal (i, i) block of the hessian matrix, it's constructed by
        using the `0e + 2e` irreps of atom i (this guarantees it being a symmetric
        matix).

        For off-diagonal (i, j), it's constructed using the 1e node features of atoms
        i and j. For each atom, we select the 3x1e irreps, separating them into
        individual 1e irrep, i.e. 1e^i_1, 1e^i_2, and 1e^i_3. Then for block (i, j)
        the prediction is obtained as

        B_ij = 1e^i_1 otimes 1e^j_1  +_ 1e^i_2 otimes 1e^j_2 + 1e^i_3 otimes 1e^j_3

        For block (j, i), it is

        B_ji = 1e^j_1 otimes 1e^i_1  +_ 1e^j_2 otimes 1e^i_2 + 1e^j_3 otimes 1e^i_3

        The sum of three 1e otimes 1e guarantees to cover any general second order
        tensor (recall one `1e otimes 1e` gives a rank 1 matrix, not general). See
        Continuum Mechanics by Tadmor -- page 45, section 2.4. In addition, B_ij and
        B_ji are transpose of each other, because 1e^i otimes 1e^j and 1e^j otimes
        1e^i are transpose of each other (they are essentially outer product of two
        vectors).

        Args:
            irreps_in:
            field:
            out_field:
        """
        super().__init__()

        self.field = field
        self.out_field = field if out_field is None else out_field

        self.init_irreps(irreps_in, required_keys_irreps_in=[field])

        # check irreps
        # note, cannot use 0e + 3x1e + 2e, which is different
        expect_irreps_in = Irreps("0e + 1e + 1e + 1e + 2e")
        assert self.irreps_in[self.field] == expect_irreps_in, (
            "expect irreps to be `0e + 1e + 1e + 1e + 2e` for the converting module; "
            f"got {self.irreps_in[self.field]}"
        )

        # extractor to separate 0e+1e+1e+1e+2e into 0e+2e and three 1e
        self.extract = Extract(
            irreps_in=expect_irreps_in,
            irreps_outs=["0e + 2e", "1e", "1e", "1e"],
            instructions=[(0, 4), (1,), (2,), (3,)],
        )

        self.tp = FullTensorProduct("1e", "1e")

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        node_feats = data[self.field]

        layout = data["hessian_off_diag_layout"]
        row_index = layout[:, 0]
        column_index = layout[:, 1]

        # extract note feats into 0e+2e and three 1e parts
        f_0e2e, f_1e_1, f_1e_2, f_1e_3 = self.extract(node_feats)

        out_ii = f_0e2e

        out_ij = (
            self.tp(f_1e_1[row_index], f_1e_1[column_index])
            + self.tp(f_1e_2[row_index], f_1e_2[column_index])
            + self.tp(f_1e_3[row_index], f_1e_3[column_index])
        )

        # output will be of two different irreps types:  0e+2e for for (i, i),
        # and 0e + 1e + 2e for (i, j).

        data["hessian_ii_block"] = out_ii
        data["hessian_ij_block"] = out_ij

        return data

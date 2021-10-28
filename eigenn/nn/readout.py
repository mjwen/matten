from typing import Dict, Optional

import torch
from e3nn.o3 import Irreps

from eigenn.nn.irreps import DataKey, ModuleIrreps
from eigenn.nn.utils import CartesianTensor


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

        self.init_irreps(irreps_in, required_keys_irreps_in=[field])

        self.field = field
        self.out_field = field if out_field is None else out_field

        self.ct = CartesianTensor(formula=formula)

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        # shallow copy so input `data` is not modified
        data = data.copy()

        value = data[self.field]
        cartesian_tensor = self.ct.to_cartesian(value)
        data[self.out_field] = cartesian_tensor

        return data

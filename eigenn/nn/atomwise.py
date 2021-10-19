from typing import Dict

import torch
from e3nn.o3 import Irreps

from eigenn.nn.irreps import DataKey, ModuleIrreps


class AtomwiseSelect(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        field: str = DataKey.NODE_FEATURES,
        out_field: str = "selected_node_features",
        mask_field: str = "node_masks",
    ):

        """
        Select the atom features (attrs) in a structure by boolean masks.

        For example, for a structure has 4 atoms, `data[mask_field] = [True,False,True,
        False]` will select the features (attrs) of atoms 0 and 2, ignoring atoms
        1 and 3.

        Args:
            irreps_in:
            field: the field from which to select the features/attrs
            out_field: the output field for the selected features/attrs
            mask_field: field of the masks
        """
        super().__init__()

        self.init_irreps(
            irreps_in=irreps_in, required_keys_irreps_in=[mask_field, field]
        )

        self.mask_field = mask_field
        self.out_field = out_field
        self.field = field

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        # shallow copy so input is not modified
        data = data.copy()

        value = data[self.field]
        masks = data[self.mask_field]
        selected = value[masks]

        data[self.out_field] = selected

        return data

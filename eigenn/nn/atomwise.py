from typing import Dict, Optional

import torch
from e3nn.o3 import Irreps

from eigenn.nn.irreps import DataKey, ModuleIrreps


class AtomwiseSelect(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        field: str = DataKey.NODE_FEATURES,
        out_field: Optional[str] = None,
        mask_field: Optional[str] = None,
    ):

        """
        Select the atom features (attrs) in a structure by boolean masks.

        For example, for a structure has 4 atoms, `data[mask_field] = [True,False,True,
        False]` will select the features (attrs) of atoms 0 and 2, ignoring atoms
        1 and 3.

        Args:
            irreps_in:
            field: the field from which to select the features/attrs
            out_field: the output field for the selected features/attrs. If `None`,
                it defaults to `field + '_selected'`.
            mask_field: field of the masks. If `None`, all atomic sites will be
                selected, corresponding to set `data[mask_field]` to `[True, True,
                True, True]` in the above example.

        Note:
            This module does not necessarily need to be a subclass of ModuleIrreps,
            since no operations on irreps are conduced. However, we use it as a proxy
            to check the existence of the fields. The idea is that if a filed is in the
            irreps_in dict, it should be in the data dict for `forward`.
            # TODO we still need to enable the check of the mask_field. For this
            #  purpose, we can add a native module update the irreps_in dict at the
            #  beginning of a model and set its irreps to `None`. Note, we do this just
            #  for consistence purpose, it is not really needed. So, we may ignore it.
        """
        super().__init__()

        required = [field]
        # if mask_field is not None:
        #     required.append(mask_field)
        self.init_irreps(irreps_in=irreps_in, required_keys_irreps_in=required)

        self.field = field
        self.out_field = out_field if out_field is not None else field + "_selected"
        self.mask_field = mask_field

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        # shallow copy so input `data` is not modified
        data = data.copy()
        value = data[self.field]

        if self.mask_field is None:
            # simply copy it over
            selected = value
        else:
            masks = data[self.mask_field]
            selected = value[masks]

        data[self.out_field] = selected

        return data
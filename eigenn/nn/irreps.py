"""
In and out irreps of a module.

This is a recreation of class GraphModuleMixin
https://github.com/mir-group/nequip/blob/main/nequip/nn/_graph_mixin.py
to make it general for different data.
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Final, Sequence, overload

import torch
import torch.nn
from e3nn.o3 import Irreps


# This is a recreation of nequip.data.AtomicDataDict
@dataclass
class DataKey:
    # type of atomic data
    Type = Dict[str, torch.Tensor]

    POSITIONS: Final[str] = "pos"
    # WEIGHTS_KEY: Final[str] = "weights"

    NODE_ATTRS: Final[str] = "node_attrs"
    NODE_FEATURES: Final[str] = "node_features"

    EDGE_INDEX: Final[str] = "edge_index"
    EDGE_CELL_SHIFT: Final[str] = "edge_cell_shift"
    EDGE_VECTORS: Final[str] = "edge_vectors"
    # EDGE_LENGTH: Final[str] = "edge_lengths"
    EDGE_ATTRS: Final[str] = "edge_attrs"
    EDGE_EMBEDDING: Final[str] = "edge_embedding"

    # CELL: Final[str] = "cell"
    # PBC: Final[str] = "pbc"

    ATOMIC_NUMBERS: Final[str] = "atomic_numbers"
    SPECIES_INDEX: Final[str] = "species_index"
    # PER_ATOM_ENERGY: Final[str] = "atomic_energy"
    # TOTAL_ENERGY: Final[str] = "total_energy"
    # FORCE: Final[str] = "forces"

    # BATCH: Final[str] = "batch"


class ModuleIrreps:
    """
    Expected input and output irreps of a module.

    subclass can implement:
      - sanity_check

    ``None`` is a valid irreps in the context for anything that is invariant but not
    well described by an ``e3nn.o3.Irreps``. An example are edge indexes in a graph,
    which are invariant but are integers, not ``0e`` scalars.

    Args:
        irreps_in:
        my_irreps_in: exact check
        required_irreps_in: only check key is present
        irreps_out:
    """

    # TODO, rename my_irreps_in -> required_exact_irreps_in
    #  require_irreps_in -> required_name_irreps_in
    def init_irreps(
        self,
        irreps_in: Dict[str, Irreps] = None,
        my_irreps_in: Dict[str, Irreps] = None,
        irreps_out: Dict[str, Irreps] = None,
        required_irreps_in: Sequence[str] = None,
    ):
        # set default
        irreps_in = {} if irreps_in is None else irreps_in
        my_irreps_in = {} if my_irreps_in is None else my_irreps_in
        irreps_out = {} if irreps_out is None else irreps_out
        required_irreps_in = [] if required_irreps_in is None else required_irreps_in

        irreps_in = _fix_irreps_dict(irreps_in)
        my_irreps_in = _fix_irreps_dict(my_irreps_in)
        irreps_out = _fix_irreps_dict(irreps_out)

        self.sanity_check(irreps_in, my_irreps_in, irreps_out, required_irreps_in)

        # Check compatibility

        # with my_irreps_in
        for k in my_irreps_in:
            if k in irreps_in and irreps_in[k] != my_irreps_in[k]:
                raise ValueError(
                    f"Input irreps {irreps_in[k]} for `{k}` is incompatible with "
                    f"this configuration {type(self)}. Should be {my_irreps_in[k]}."
                )

        # with required_irreps_in
        for k in required_irreps_in:
            if k not in irreps_in:
                raise ValueError(
                    f"This configuration {type(self)} requires `{k}` in `irreps_in`."
                )

        # Save stuff
        self._irreps_in = irreps_in

        # The output irreps of any graph module are whatever inputs it has,
        # overwritten with whatever outputs it has.
        self._irreps_out = irreps_in.copy()
        self._irreps_out.update(irreps_out)

    @property
    def irreps_in(self):
        return self._irreps_in

    @property
    def irreps_out(self):
        return self._irreps_out

    def sanity_check(
        self,
        irreps_in=None,
        my_irreps_in=None,
        irreps_out=None,
        required_irreps_in=None,
    ):
        """
        Check the input of the class.
        """

        # positions are always 1o and should always be present
        pos = DataKey.POSITIONS
        if pos in irreps_in and irreps_in[pos] != Irreps("1x1o"):
            raise ValueError(f"Positions must have irreps 1o, got `{irreps_in[pos]}`")
        irreps_in[pos] = Irreps("1o")

        # edges are always None and should always be present
        edge_index = DataKey.EDGE_INDEX
        if edge_index in irreps_in and irreps_in[edge_index] is not None:
            raise ValueError(
                f"Edge indexes must have irreps `None`, got `{irreps_in[edge_index]}`"
            )
        irreps_in[edge_index] = None


class Sequential(ModuleIrreps, torch.nn.Sequential):
    """
    This is the same as torch.nn.Sequential, with additional check on irreps
    compatibility between consecutive modules.
    """

    @overload
    def __init__(self, *args: ModuleIrreps) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, ModuleIrreps]") -> None:
        ...

    def __init__(self, *args):

        # dict input
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            module_dict = args[0]
            module_list = list(module_dict.values())
        # sequence input
        else:
            module_list = list(args)
            module_dict = OrderedDict(
                (f"{m.__class__.__name__}_{i}", m) for i, m in enumerate(module_list)
            )

        # check in/out irreps compatibility
        for i, (m1, m2) in enumerate(zip(module_list, module_list[1:])):
            if not _check_irreps_compatible(m1.irreps_out, m2.irreps_in):
                raise ValueError(
                    f"Output irreps of module {i} `{m1.__class__.__name__}`: "
                    f"{m1.irreps_out}` is incompatible with input irreps of module {i+1} "
                    f"`{m2.__class__.__name__}`: {m2.irreps_in}."
                )

        self.init_irreps(
            irreps_in=module_list[0].irreps_in, irreps_out=module_list[-1].irreps_out
        )

        super().__init__(module_dict)


# copied from nequip:
# https://github.com/mir-group/nequip/blob/main/nequip/data/AtomicData.py
def _fix_irreps_dict(irreps: Dict[str, Irreps]) -> Dict[str, Irreps]:
    """
    Fix irreps.

      - convert string representation to object
      - deal with None. ``None`` is a valid irreps in the context for anything that
        is invariant but not well described by an ``e3nn.o3.Irreps``. An example are
        edge indexes in a graph, which are invariant but are integers, not ``0e``
        scalars.
    """
    special_irreps = [None]
    irreps = {k: (i if i in special_irreps else Irreps(i)) for k, i in irreps.items()}

    return irreps


# copied from nequip:
# https://github.com/mir-group/nequip/blob/main/nequip/data/AtomicData.py
def _check_irreps_compatible(ir1: Dict[str, Irreps], ir2: Dict[str, Irreps]):
    return all(ir1[k] == ir2[k] for k in ir1 if k in ir2)

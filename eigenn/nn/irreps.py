"""
In and out irreps of a module.

This is a recreation of class GraphModuleMixin
https://github.com/mir-group/nequip/blob/main/nequip/nn/_graph_mixin.py
to make it general for different data.
"""
from dataclasses import dataclass
from typing import Dict, Final, List, Sequence

import torch.nn
from e3nn.o3 import Irrep, Irreps


# This is a recreation of nequip.data.AtomicDataDict
@dataclass
class DataKey:
    # type of atomic data
    Type = Dict[str, torch.Tensor]

    # positions of nodes in 3D space
    POSITIONS: Final[str] = "pos"
    # WEIGHTS_KEY: Final[str] = "weights"

    # attributes on node; fixed
    NODE_ATTRS: Final[str] = "node_attrs"

    # features on node, e.g. embedding of atomic species; learnable
    NODE_FEATURES: Final[str] = "node_features"

    EDGE_INDEX: Final[str] = "edge_index"
    EDGE_CELL_SHIFT: Final[str] = "edge_cell_shift"
    EDGE_VECTORS: Final[str] = "edge_vectors"
    # EDGE_LENGTH: Final[str] = "edge_lengths"

    # spherical part of edge vector (i.e. expansion of the unit displacement vector
    # on spherical harmonics); fixed
    EDGE_ATTRS: Final[str] = "edge_attrs"

    # TDDO change this to EDGE_FEATURES?
    # radial part of the edge vector (i.e. distance between atoms), learnable
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
      - REQUIRED_KEY_IRREPS_IN
      - OPTIONAL_EXACT_IRREPS_IN
      - fix_irreps_in

    ``None`` is a valid irreps in the context for anything that is invariant but not
    well described by an ``e3nn.o3.Irreps``. An example are edge indexes in a graph,
    which are invariant but are integers, not ``0e`` scalars.

    Args:
        irreps_in: input irreps, availables keys in `DataKey`
        irreps_out: a dict of output irreps, availables keys in `DataKey`. If a string,
            it should be a key in irreps_in, and then irreps_out will be
            {key: irreps_in[key]}
        required_keys_irreps_in: gives the required keys should be present in irreps_in.
            This only requires the irreps is given in `irreps_in`; does not specify
            what the irreps look like.
        optional_exact_irreps_in: for irreps in this dict, if it given in `irreps_in`,
            they two should match (i.e. be the same). It's not required that irreps
            specified in this dict has to be present in `irreps_in`.

    Attrs:
        REQUIRED_KEY_IRREPS_IN
        OPTIONAL_EXACT_IRREPS_IN
    """

    REQUIRED_KEYS_IRREPS_IN = None
    OPTIONAL_EXACT_IRREPS_IN = None

    def init_irreps(
        self,
        irreps_in: Dict[str, Irreps] = None,
        irreps_out: Dict[str, Irreps] = None,
        *,
        required_keys_irreps_in: Sequence[str] = None,
        optional_exact_irreps_in: Dict[str, Irreps] = None,
    ):

        # input irreps
        irreps_in = {} if irreps_in is None else irreps_in
        irreps_in = _fix_irreps_dict(irreps_in)
        irreps_in = self.fix_irreps_in(irreps_in)

        # output irreps
        if irreps_out is None:
            irreps_out = {}
        elif isinstance(irreps_out, str):
            assert (
                irreps_out in irreps_in
            ), f"`irreps_in` does not contain key for `irreps_out = {irreps_out}`"
            irreps_out = {irreps_out, irreps_in[irreps_out]}
        irreps_out = _fix_irreps_dict(irreps_out)

        # required input irreps
        required = (
            [] if self.REQUIRED_KEYS_IRREPS_IN is None else self.REQUIRED_KEYS_IRREPS_IN
        )
        if required_keys_irreps_in is not None:
            required += list(required_keys_irreps_in)

        # optional exact irreps
        optional = (
            {}
            if self.OPTIONAL_EXACT_IRREPS_IN is None
            else self.OPTIONAL_EXACT_IRREPS_IN
        )
        if optional_exact_irreps_in is not None:
            optional.update(optional_exact_irreps_in)
        optional = _fix_irreps_dict(optional)

        # Check compatibility

        # check optional_exact_irreps_in
        for k in optional:
            if k in irreps_in and irreps_in[k] != optional[k]:
                raise ValueError(
                    f"Input irreps {irreps_in[k]} for `{k}` is incompatible with this "
                    f"configuration {type(self)}. Should be {optional[k]}"
                )

        # check required_keys_irreps_in
        for k in required:
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

    def fix_irreps_in(self, irreps_in: Dict[str, Irreps]) -> Dict[str, Irreps]:
        """
        Fix the input irreps.
        """
        irreps_in = irreps_in.copy()

        # positions are always 1o and should always be present in
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

        return irreps_in


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


def _check_irreps_type(irreps: Irreps, allowed: List[Irrep]) -> bool:
    """
    Check the irreps only contains the allowed type.

    This only checks the type (degree and parity), not the multiplicity.

    Args:
        irreps: the irreps to check
        allowed: allowed irrep (degree and parity), e.g. ['0e', '1o']
    """
    irreps = Irreps(irreps)
    allowed = [Irrep(i) for i in allowed]

    for m, ir in irreps:
        if ir not in allowed:
            return False

    return True

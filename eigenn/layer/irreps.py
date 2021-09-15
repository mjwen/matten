"""
In and out irreps of a module.

This is a recreation of class GraphModuleMixin
https://github.com/mir-group/nequip/blob/main/nequip/nn/_graph_mixin.py
to make it general for different data.
"""
from dataclasses import dataclass
from typing import Dict, Final, Sequence

from e3nn.o3 import Irreps


@dataclass
class DataKey:
    POSITIONS: Final[str] = "pos"
    EDGE_INDEX: Final[str] = "edge_index"


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

        irreps_in = self._fix_irreps_dict(irreps_in)
        my_irreps_in = self._fix_irreps_dict(my_irreps_in)
        irreps_out = self._fix_irreps_dict(irreps_out)

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
        assert pos in irreps_in, "Positions (1o) should be present in irreps_in."
        if irreps_in[pos] != Irreps("1x1o"):
            raise ValueError(f"Positions must have irreps 1o, got `{irreps_in[pos]}`")

        # edges are always None and should always be present
        edge_index = DataKey.EDGE_INDEX
        assert (
            edge_index in irreps_in
        ), "Edge indexes (None) should be present in irreps_in"
        if irreps_in[edge_index] is not None:
            raise ValueError(
                f"Edge indexes must have irreps `None`, got `{irreps_in[edge_index]}`"
            )

    @staticmethod
    def _fix_irreps_dict(irreps: Dict[str, Irreps]) -> Dict[str, Irreps]:
        """
        Fix irreps:
          - convert string representation to object
          - deal with None. ``None`` is a valid irreps in the context for anything that
            is invariant but not well described by an ``e3nn.o3.Irreps``. An example are
            edge indexes in a graph, which are invariant but are integers, not ``0e``
            scalars.
        """
        special_irreps = [None]
        irreps = {
            k: (i if i in special_irreps else Irreps(i)) for k, i in irreps.items()
        }

        return irreps
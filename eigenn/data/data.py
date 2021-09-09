import itertools
import warnings
from typing import Dict, List
from pymatgen.core.structure import Structure

import ase
import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data

from eigenn.core.configuration import Configuration
from eigenn.typing import PBC, IntVector, Vector

DTYPE = torch.get_default_dtype()


class DataPoint(Data):
    """
    Base class of a graph data point (be it a molecule or a crystal).

    Args:
        pos: coords of the atoms
        node_attrs: extra node attributes (e.g. species of atoms). These are different
            from `x` in that they are not used as input for the model.
        edge_index: 2D array (2, num_edges). edge index.
        edge_cell_shift: 2D array (num_edges, 3). which periodic image of the target
            point each edge goes to, relative to the source point. Used when periodic
            boundary conditions is effective (e.g. for crystals).
        cell: 3D array (1, 3, 3). The periodic cell for ``edge_cell_shift`` as the three
            triclinic cell vectors. Necessary only when ``edge_cell_shift`` is not None.
        x: input to the model, i.e. initial node features
        y: reference value for the output of the model, e.g. DFT energy, forces
        kwargs: extra property of a data point, e.g. prediction output such as energy
            and forces.
    """

    def __init__(
        self,
        *,
        pos: List[Vector],
        edge_index: npt.ArrayLike,
        x: Dict[str, npt.ArrayLike],
        y: Dict[str, npt.ArrayLike],
        edge_cell_shift: List[IntVector] = None,
        cell: npt.ArrayLike = None,
        **kwargs,
    ):

        # convert to tensors
        pos = torch.as_tensor(pos, dtype=DTYPE)
        edge_index = torch.as_tensor(edge_index, dtype=torch.int64)
        edge_cell_shift = (
            torch.as_tensor(edge_cell_shift, dtype=torch.int64)
            if edge_cell_shift
            else None
        )
        cell = torch.as_tensor(cell, dtype=DTYPE) if cell else None

        # check shape
        num_nodes = pos.shape[0]
        num_edges = edge_index.shape[1]

        assert pos.shape[1] == 3
        assert edge_index.shape[0] == 2

        if edge_cell_shift is not None:
            assert edge_cell_shift.shape == [num_edges, 3]
            assert cell is None, "both `edge_cell_shift` and `cell` should be provided"
        if cell is not None:
            assert cell.shape == (1, 3, 3)
            assert edge_cell_shift is None, (
                "both `edge_cell_shift` and `cell` should " "be provided"
            )

        # convert input and output to tensors
        x = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in x.items()}
        y = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in y.items()}
        self.sanity_check()

        # pyG Data only accepts tensor as input, not dict.
        # Here, we convert each key in x (y) by prepending `x_` (`y_`) and assign it as
        # an attribute.
        for k, v in x.items():
            new_k = f"x_{k}"
            assert (
                new_k not in kwargs
            ), f"Cannot assign input `{k}` in the `x` dict as Data attribute."
            kwargs[new_k] = v

        for k, v in y.items():
            new_k = f"y_{k}"
            assert (
                new_k not in kwargs
            ), f"Cannot assign input `{k}` in the `y` dict as Data attribute."
            kwargs[new_k] = v

        super().__init__(
            pos=pos,
            edge_index=edge_index,
            edge_cell_shift=edge_cell_shift,
            cell=cell,
            num_nodes=num_nodes,
            **kwargs,
        )

    def sanity_check(self):
        """
        Check the shape of the input x and output y.
        """
        pass


class Molecule(DataPoint):
    """
    Molecule graph data point, without the notation of supercell.
    """

    def __init__(
        self,
        *,
        pos: List[Vector],
        edge_index: npt.ArrayLike,
        x: Dict[str, npt.ArrayLike],
        y: Dict[str, npt.ArrayLike],
        **kwargs,
    ):

        super().__init__(pos=pos, edge_index=edge_index, x=x, y=y, **kwargs)

    @classmethod
    def from_configuration(
        cls,
        config: Configuration,
        x: Dict[str, npt.ArrayLike],
        y: Dict[str, npt.ArrayLike],
        edges_from: str = "bonds",
        **kwargs,
    ):
        """
        Create a molecule from :class:`eigenn.core.configuration.Configuration`.

        Args:
            config: a configuration
            x:
            y:
            edges_from: the method used to determine the edges. Options are:
                - ``bonds``, edges created from the bonds between the atoms.
                - ``complete``, edges created between every pair of atoms.
                - ``<n>_hop``, edges created from n (n=1,2,...) hop bond distance.
                    This is an intermediate between ``bonds`` and ``complete``:
                    n=1 is the same as ``bonds`` and a very large n is most likely the
                    same as ``complete``.
        """
        if edges_from == "bonds":
            bonds = config.get_bonds()
            edge_index = list(zip(*bonds))
        elif edges_from == "complete":
            natoms = config.get_num_atoms()
            edge_index = list(zip(*itertools.product(natoms, natoms)))
        elif "_hop" in edges_from:
            n = int(edges_from.split("_")[0])
            raise NotImplementedError
        else:
            raise DataError(f"Not supported `edges_from`: `{edges_from}`")

        return cls(pos=config.coords, edge_index=edge_index, x=x, y=y, **kwargs)


class Crystal(DataPoint):
    """
    Crystal graph data point, with the notation of supercell.
    """

    @classmethod
    def from_points(
        cls,
        *,
        pos: List[Vector],
        cell: npt.ArrayLike,
        pbc: PBC,
        r_cut: float,
        x: Dict[str, npt.ArrayLike],
        y: Dict[str, npt.ArrayLike],
        **kwargs,
    ):
        """
        Create a crystal from a set of points.

        Args:
            pos:
            cell:
            pbc:
            r_cut: neighbor cutoff distance
            x:
            y:
            **kwargs:
        """

        edge_index, edge_cell_shift, cell = neighbor_list_and_relative_vec(
            pos=pos,
            r_max=r_cut,
            self_interaction=False,
            strict_self_interaction=True,
            cell=cell,
            pbc=pbc,
        )

        return cls(
            pos=pos,
            edge_index=edge_index,
            x=x,
            y=y,
            edge_cell_shift=edge_cell_shift,
            cell=cell,
            **kwargs,
        )

    @classmethod
    def from_configuration(
        cls,
        config: Configuration,
        r_cut: float,
        x: Dict[str, npt.ArrayLike],
        y: Dict[str, npt.ArrayLike],
        **kwargs,
    ):
        """
        Create a molecule from :class:`eigenn.core.configuration.Configuration`.
        """
        return cls.from_points(
            pos=config.coords,
            cell=config.cell,
            pbc=config.pbc,
            r_cut=r_cut,
            x=x,
            y=y,
            **kwargs,
        )

    @classmethod
    def from_pymatgen(
        cls,
        struct: Structure,
        r_cut: float,
        x: Dict[str, npt.ArrayLike],
        y: Dict[str, npt.ArrayLike],
        **kwargs,
    ):
        return cls.from_points(
            pos=struct.coords,
            cell=struct.cell,
            pbc=[True, True, True],
            r_cut=r_cut,
            x=x,
            y=y,
            **kwargs,
        )


# This function is copied from nequip.data.AtomicData
def neighbor_list_and_relative_vec(
    pos,
    r_max,
    self_interaction=False,
    strict_self_interaction=True,
    cell=None,
    pbc=False,
):
    """
    Create neighbor list (``edge_index``) and relative vectors (``edge_attr``) based on
    radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).
    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`
    If the input positions are a tensor with ``requires_grad == True``,
    the output displacement vectors will be correctly attached to the inputs
    for autograd.
    All outputs are Tensors on the same device as ``pos``; this allows future
    optimization of the neighbor list on the GPU.

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor,
            must be on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions.
            Ignored if ``pbc == False``.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the
            three cell dimensions.
        self_interaction (bool): Whether or not to include same periodic image
            self-edges in the neighbor list.
        strict_self_interaction (bool): Whether to include *any* self interaction edges
            in the graph, even if the two instances of the atom are in different periodic
            images. Defaults to True, should be True for most applications.
    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_cell_shift (torch.tensor shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
        cell (torch.Tensor [3, 3]): the cell as a tensor on the correct device.
            Returned only if cell is not None.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    # Either the position or the cell may be on the GPU as tensors
    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().cpu().numpy()
        out_device = pos.device
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    # Right now, GPU tensors require a round trip
    if out_device.type != "cpu":
        warnings.warn(
            "Currently, neighborlists require a round trip to the CPU. "
            "Please pass CPU tensors if possible."
        )

    # Get a cell on the CPU no matter what
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
        cell_tensor = cell.to(device=out_device, dtype=out_dtype)
    elif cell is not None:
        temp_cell = np.asarray(cell)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)
    else:
        # ASE will "complete" this correctly.
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)

    # ASE dependent part
    temp_cell = ase.geometry.complete_cell(temp_cell)

    first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=float(r_max),
        self_interaction=strict_self_interaction,  # we want edges from atom to itself in different periodic images!
        use_scaled_positions=False,
    )

    # Eliminate true self-edges that don't cross periodic boundaries
    if not self_interaction:
        bad_edge = first_idex == second_idex
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise ValueError(
                "After eliminating self edges, no edges remain in this system."
            )
        first_idex = first_idex[keep_edge]
        second_idex = second_idex[keep_edge]
        shifts = shifts[keep_edge]

    # Build output:
    edge_index = torch.vstack(
        (torch.LongTensor(first_idex), torch.LongTensor(second_idex))
    ).to(device=out_device)

    shifts = torch.as_tensor(
        shifts,
        dtype=out_dtype,
        device=out_device,
    )
    return edge_index, shifts, cell_tensor


class DataError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg

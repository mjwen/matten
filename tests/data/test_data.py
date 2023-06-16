from pathlib import Path

import ase.io
import numpy as np
from torch_geometric.data import Batch

from matten.data.data import DataPoint

TEST_FILE_DIR = Path(__file__).resolve().parents[1].joinpath("test_files")


def test_AtomicData():
    natoms = 4
    pos = np.random.randn(natoms, 3)
    edge_index = np.random.randint(0, natoms - 1, (2, 6))
    x = {"species": np.asarray([0, 0, 1, 4]), "coords": pos}
    y = {"energy": np.asarray([0.1]), "forces": np.random.randn(natoms, 3)}

    data = DataPoint(pos, edge_index=edge_index, x=x, y=y)

    batch = Batch.from_data_list([data, data])

    print(data)
    print("\n\n\n")
    print(batch)


def test_pmg_mol_graph():
    file = TEST_FILE_DIR.joinpath("mol.xyz")
    atoms = ase.io.read(file, format="extxyz")
    pos = atoms.positions
    species = atoms.symbols

    edges, num_neigh = pmg_mol_graph(pos, species)

    ref_edges = np.asarray(
        [
            [0, 6],
            [0, 1],
            [0, 4],
            [0, 5],
            [1, 2],
            [1, 0],
            [2, 7],
            [2, 3],
            [2, 1],
            [3, 8],
            [3, 4],
            [3, 2],
            [4, 0],
            [4, 3],
            [5, 0],
            [6, 0],
            [7, 2],
            [8, 3],
        ]
    ).T
    ref_num_neigh = [4, 2, 3, 3, 2, 1, 1, 1, 1]

    assert np.array_equal(edges, ref_edges)
    assert num_neigh == ref_num_neigh

import numpy as np
from torch_geometric.data import Batch

from eigenn.data.data import DataPoint


def test_AtomicData():

    natoms = 4
    pos = np.random.randn(natoms, 3)
    edge_index = np.random.randint(0, natoms - 1, (2, 6))
    x = {"species": [0, 0, 1, 4], "coords": np.random.randn(natoms, 3)}
    y = {"energy": [0.1], "forces": np.random.randn(natoms, 3)}

    data = DataPoint(pos, edge_index=edge_index, x=x, y=y)

    batch = Batch.from_data_list([data, data])

    print(data)
    print("\n\n\n")
    print(batch)

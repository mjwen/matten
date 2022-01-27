import itertools
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ase.data
import ase.io
import numpy as np
import torch

from eigenn.data.data import Crystal, Molecule
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset


class HessianDataset(InMemoryDataset):
    """
    Hessian dataset for small molecules.

    Args:
         filename:
         root:
    """

    def __init__(self, filename: str, root: Union[str, Path] = "."):
        super().__init__(
            filenames=[filename], root=root, processed_dirname=f"processed"
        )

    def get_data(self):

        filepath = self.raw_paths[0]
        configs = ase.io.read(filepath, index=":")

        molecules = []
        for i, conf in enumerate(configs):
            try:
                # Hessian is a 3N by 3N matrix. But since each molecule has different
                # number of atoms N, directly passing this to dataloader should not
                # work (cannot batch). So we reshape it to a N^2 by 3 by 3 matrix.
                N = len(conf)
                hessian = conf.info["hessian"].reshape(N, 3, N, 3)
                hessian = np.swapaxes(hessian, 1, 2)
                hessian = hessian.reshape(-1, 3, 3)

                m = Molecule(
                    pos=conf.positions,
                    edge_index=self.fully_connected_graph(N),
                    x=None,
                    y={"hessian": hessian},
                    atomic_numbers=conf.get_atomic_numbers(),
                )
                molecules.append(m)
            except Exception as e:
                warnings.warn(
                    f"Failed converting configuration {i}: {str(e)}. Skip it."
                )

        return molecules

    @staticmethod
    def fully_connected_graph(N: int) -> np.ndarray:
        """
        Edge index of a fully-connected graph.

        Args:
            N: number of atoms

        Returns:
            edge index, shape (2, N).
        """
        edge_index = np.asarray(list(zip(*itertools.product(range(N), range(N)))))

        return edge_index


if __name__ == "__main__":
    filename = "ani1_CHO_0-1000_hessian_small.xyz"
    root = "/Users/mjwen/Documents/Dataset/xiaowei_hessian"
    dataset = HessianDataset(filename=filename, root=root)

    for m in dataset:
        print(m)
        break

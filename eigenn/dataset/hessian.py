import itertools
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ase.data
import ase.io
import numpy as np
import torch

from eigenn.data.data import Molecule
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

                # How the N^2 part of the hessian is lay out.
                # First column gives the row index of the hessian matrix (without
                # considering the 3) and second column gives the column index.
                # As an example, for a system with 2 atoms, this will be
                # [[0, 0],
                #  [0, 1],
                #  [1, 0],
                #  [1, 1]]
                # meaning that the first 3x3 matrix in the hessian is the (0,
                # 0) submatrix, and the second is the (0,1) submatrix...
                #
                # We use this row based stuff (not something similar to edge_index) for
                # easy batching.
                hessian_layout = np.asarray(
                    list(zip(*itertools.product(range(N), range(N))))
                ).T

                # number of atoms of the config, repeated N^2 times, one for each
                # 3x3 block
                hessian_natoms = N * np.ones(N * N)

                edge_index = self.fully_connected_graph(N)
                num_neigh = [N - 1 for _ in range(N)]

                m = Molecule(
                    pos=conf.positions,
                    edge_index=edge_index,
                    x=None,
                    y={
                        "hessian": hessian,
                        "hessian_layout": hessian_layout,
                        "hessian_natoms": hessian_natoms,
                    },
                    atomic_numbers=conf.get_atomic_numbers(),
                    num_neigh=num_neigh,
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


class HessianDataMoldule(BaseDataModule):
    """
    Will search for files at, e.g. `root/<trainset_filename>`.
    """

    def __init__(
        self,
        trainset_filename: str,
        valset_filename: str,
        testset_filename: str,
        *,
        root: Union[str, Path] = ".",
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.root = root

        super().__init__(
            trainset_filename,
            valset_filename,
            testset_filename,
            state_dict_filename=state_dict_filename,
            restore_state_dict_filename=restore_state_dict_filename,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_data = HessianDataset(self.trainset_filename, self.root)
        self.val_data = HessianDataset(self.valset_filename, self.root)
        self.test_data = HessianDataset(self.testset_filename, self.root)

    def get_to_model_info(self) -> Dict[str, Any]:
        atomic_numbers = set()
        num_neigh = []
        for data in self.train_dataloader():
            a = data.atomic_numbers.tolist()
            atomic_numbers.update(a)
            num_neigh.append(data.num_neigh)

        # .item to convert to float so that lightning cli can save it to yaml
        average_num_neighbors = torch.mean(torch.cat(num_neigh)).item()

        return {
            "allowed_species": tuple(atomic_numbers),
            "average_num_neighbors": average_num_neighbors,
        }


if __name__ == "__main__":
    filename = "ani1_CHO_0-1000_hessian_small.xyz"
    root = "/Users/mjwen/Documents/Dataset/xiaowei_hessian"
    dm = HessianDataMoldule(
        trainset_filename=filename,
        valset_filename=filename,
        testset_filename=filename,
        root=root,
    )
    dm.setup()
    info = dm.get_to_model_info()
    print(info)

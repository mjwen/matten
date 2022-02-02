import itertools
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ase.data
import ase.io
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset, HeteroData

from eigenn.data.data import Molecule
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset


class HessianDataset(InMemoryDataset):
    """
    Hessian dataset for small molecules.

    Args:
         filename:
         root:
         edge_strategy: `complete` | `pmg_mol_graph`
    """

    def __init__(
        self,
        filename: str,
        root: Union[str, Path] = ".",
        reuse=True,
        edge_strategy: str = "pmg_mol_graph",
    ):
        self.edge_strategy = edge_strategy

        super().__init__(
            filenames=[filename],
            root=root,
            processed_dirname=f"processed_edge_strategy-{self.edge_strategy}",
            reuse=reuse,
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
                # 0) sub-matrix, and the second is the (0,1) sub-matrix...
                #
                # We use this row based stuff (not something similar to edge_index) for
                # easy batching.
                hessian_layout = np.asarray(
                    list(zip(*itertools.product(range(N), repeat=2)))
                ).T

                # number of atoms of the config, repeated N^2 times, one for each
                # 3x3 block
                hessian_natoms = N * np.ones(N * N)

                m = Molecule.with_edge_strategy(
                    pos=conf.positions,
                    x=None,
                    y={
                        "hessian": hessian,
                        "hessian_layout_raw": hessian_layout,
                        "hessian_natoms": hessian_natoms,
                    },
                    strategy=self.edge_strategy,
                    atomic_numbers=conf.get_atomic_numbers(),
                )
                molecules.append(m)

            except Exception as e:
                warnings.warn(
                    f"Failed converting configuration {i}: {str(e)}. Skip it."
                )

        return molecules


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
        reuse: bool = True,
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.root = root

        super().__init__(
            trainset_filename,
            valset_filename,
            testset_filename,
            reuse=reuse,
            state_dict_filename=state_dict_filename,
            restore_state_dict_filename=restore_state_dict_filename,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_data = HessianDataset(
            self.trainset_filename, self.root, reuse=self.reuse
        )
        self.val_data = HessianDataset(
            self.valset_filename, self.root, reuse=self.reuse
        )
        self.test_data = HessianDataset(
            self.testset_filename, self.root, reuse=self.reuse
        )

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

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, **self.loader_kwargs)

    def val_dataloader(self):
        loader_kwargs = self.loader_kwargs.copy()
        loader_kwargs.pop("shuffle", None)
        return DataLoader(dataset=self.val_data, **loader_kwargs)

    def test_dataloader(self):
        loader_kwargs = self.loader_kwargs.copy()
        loader_kwargs.pop("shuffle", None)
        return DataLoader(dataset=self.test_data, **loader_kwargs)


#
# Collater and DataLoader copied from PyG.
# We need to use `hessian_layout` to select node features. In a sense, it is similar
# to how edge_index should be treated when batching: i.e. the index of the next graph
# should be added by the total number of edges in the previous graph. However,
# since we provide it as an attribute in y, this will not be done by the default
# collate function. Here, we modify it to deal with this.
#
class Collater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data) or isinstance(elem, HeteroData):

            # @@@ NOTE, add Data.y.hessian_layout
            i = 0
            for d in batch:
                # should use different name to not overwrite hessian_layout_raw
                d["y"]["hessian_layout"] = d["y"]["hessian_layout_raw"] + i
                i += d.num_nodes
            # @@@

            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError("DataLoader found invalid type: {}".format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`[]`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, List[Data], List[HeteroData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: List[str] = [],
        exclude_keys: List[str] = [],
        **kwargs,
    ):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )


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

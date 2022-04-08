import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ase.data
import ase.io
import torch
import torch.utils.data
from e3nn.io import CartesianTensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset, HeteroData
from torchtyping import TensorType

from eigenn.data.data import Molecule
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset


class HessianDataset(InMemoryDataset):
    """
    Hessian dataset for small molecules.

    Args:
        filename:
        root:
        reuse: whether to reuse the preprocessed data.
        edge_strategy: `complete` | `pmg_mol_graph`
    """

    def __init__(
        self,
        filename: str,
        *,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        edge_strategy: str = "pmg_mol_graph",
        output_format: str = "irreps",
        output_formula: str = "ij=ij",  # TODO delete this, not used
    ):
        self.edge_strategy = edge_strategy
        self.output_format = output_format
        self.output_formula = output_formula

        super().__init__(
            filenames=[filename],
            root=root,
            processed_dirname=f"processed_edge_strategy-{self.edge_strategy}",
            reuse=reuse,
        )

    def get_data(self):

        filepath = self.raw_paths[0]
        configs = ase.io.read(filepath, index=":")

        # convert to irreps tensor is necessary
        converter_diag = CartesianTensor(formula="ij=ji")
        converter_off_diag = CartesianTensor(formula="ij=ij")

        molecules = []
        for i, conf in enumerate(configs):

            try:
                # Hessian is a 3N by 3N matrix. But since each molecule has different
                # number of atoms N, directly passing this to dataloader should not
                # work (cannot batch). So we reshape it to a N^2 by 3 by 3 matrix.
                N = len(conf)
                hessian = torch.as_tensor(conf.info["hessian"], dtype=torch.float32)
                hessian = hessian.reshape(N * 3, N * 3)
                diag, off_diag, off_diag_layout = separate_diagonal_blocks(hessian)

                if self.output_format == "irreps":
                    diag = converter_diag.from_cartesian(diag)
                    off_diag = converter_off_diag.from_cartesian(off_diag)

                m = Molecule.with_edge_strategy(
                    pos=conf.positions,
                    x=None,
                    y={
                        "hessian_diag": diag,
                        "hessian_off_diag": off_diag,
                        "hessian_off_diag_layout_raw": off_diag_layout,
                        "hessian_natoms": torch.tensor([N]),
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


class HessianDataModule(BaseDataModule):
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
        output_format: str = "cartesian",
        output_formula: str = "ij=ij",
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.root = root
        self.output_format = output_format
        self.output_formula = output_formula

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
            self.trainset_filename,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
        )
        self.val_data = HessianDataset(
            self.valset_filename,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
        )
        self.test_data = HessianDataset(
            self.testset_filename,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
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
# We need to use `hessian_off_diag_layout` to select node features. In a sense,
# it is similar
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

            # add Data.y.hessian_off_diag_layout, this is similar to edge index,
            # should increase by the number of atoms in each molecule.
            #
            i = 0
            for d in batch:
                # should use different name to not overwrite hessian_off_diag_layout_raw
                d["y"]["hessian_off_diag_layout"] = (
                    d["y"]["hessian_off_diag_layout_raw"] + i
                )
                i += d.num_nodes

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


def symmetrize_hessian(
    H: TensorType["nblocks", 3, 3], natoms: List[int]  # noqa: F821
) -> torch.Tensor:
    """
    Symmetrize a Hessian matrix by H = (H + H^T)/2, where T denotes matrix transpose.

    This can deal with batched cases.

    Args:
        H: The batched Hessian to symmetrize. For a list of molecules
            with number of atoms N1, N2, ... N_n, nblocks = N1^2 + N2^2 + ... + N_n^2.
            Also, this assumes each for each molecule, the N1^2 3x3 blocks is stacked
            according to
            [(0, 0),
             (0, 1),
             ...
             (0, N1-1)
             (1, 0),
             ...
             (N1-1, N1-1)
             ]
            which is the same as HessianDataset.get_data().
        natoms: number of atoms for individual molecules in the batch.
    """
    H_by_mol = torch.split(H, [i**2 for i in natoms], dim=0)

    # This can be slow, do we have better ways to do it?
    sym_H = []
    for x, n in zip(H_by_mol, natoms):
        x = x.reshape(n, n, 3, 3)
        x = torch.swapaxes(x, 1, 2)
        x = x.reshape(n * 3, n * 3)
        x = (x + x.T) / 2
        x = x.reshape(n, 3, n, 3)
        x = torch.swapaxes(x, 1, 2)
        x = x.reshape(-1, 3, 3)
        sym_H.append(x)

    return torch.cat(sym_H)


def separate_diagonal_blocks(
    H: TensorType["N*3", "N*3"],  # noqa: F821
) -> Tuple[
    TensorType["N", 3, 3],  # noqa: F821
    TensorType["N*N-N", 3, 3],  # noqa: F821
    TensorType["N*N-N", 2],  # noqa: F821
]:
    """
    Separate the 3N by 3N Hessian matrix into diagonal and off-diagonal 3 by 3 blocks.

    Args:
        H: the hessian matrix

    Returns:
        diagonal: Diagonal blocks of the Hessian matrix
        off_diagonal: off-diagonal blocks of the Hessian matrix.
        off_diagonal_layout: How the off diagonal part of the Hessian is layed out.
            I.e. the mapping between the off-diagonal blocks and this returned value is
            (0, 1) -> 0, (0, 2)-> 1, ... (0, N-1) -> N-2,
            (1, 0)-> N-1, (1, 2)-> N ...

            First column gives the row index of the hessian matrix (without
            considering the 3) and second column gives the column index.
    """
    N = len(H) // 3  # number of atoms

    diag_idx = [i * (N + 1) for i in range(N)]
    offdiag_idx = sorted(set(range(N * N)) - set(diag_idx))

    H = H.reshape(N, 3, N, 3).swapaxes(1, 2).reshape(N * N, 3, 3)
    diag = H[diag_idx]
    offdiag = H[offdiag_idx]

    offdiag_layout = torch.as_tensor(
        [[i, j] for i in range(N) for j in range(N) if i != j]
    )

    return diag, offdiag, offdiag_layout


if __name__ == "__main__":
    filename = "ani1_CHO_0-1000_hessian_small.xyz"
    root = "/Users/mjwen/Documents/Dataset/xiaowei_hessian"
    dm = HessianDataModule(
        trainset_filename=filename,
        valset_filename=filename,
        testset_filename=filename,
        root=root,
        reuse=False,
        output_format="irreps",
        output_formula="ij=ij",
        loader_kwargs={"batch_size": 2},
    )
    dm.setup()
    info = dm.get_to_model_info()
    print("to_model_info", info)

    for graph in dm.train_dataloader():
        print("graph", graph)
        print("graph.batch", graph.batch)
        for key, v in graph["y"].items():
            print(f"{key}: {v}")

        break

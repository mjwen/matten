import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torchtyping import TensorType

from eigenn.data.data import Crystal
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset
from eigenn.data.transform import ScalarNormalize


class StructureScalarDataset(InMemoryDataset):
    """
    A dataset that is intended for mapping a pymatgen structure to scalar properties.

    Args:
        filename: name of data file.
        r_cut: neighbor cutoff distance, in unit Angstrom.
        target_names: name of the properties to be used as target.
        root: root directory that stores the input and processed data.
        reuse: whether to reuse the preprocessed data.
        log_target: whether to log transform the targets.
        compute_dataset_statistics: callable to compute dataset statistics. Do not
            compute if `None`. Note, this is different from `normalize_target` below.
            This only determines whether to compute the statistics of the target of a
            dataset, and will generate a file named `dataset_statistics.pt` is $CWD
            if a callable is provided. Whether to use dataset statistics to do
            normalization is determined by `normalize_target`.
        normalize_target: whether to normalize the target.
    """

    def __init__(
        self,
        filename: str,
        r_cut: float,
        target_names: List[str],
        *,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        log_target: bool = True,
        compute_dataset_statistics: Callable = None,
        normalize_target: bool = False,
        normalizer_kwargs: Dict[str, Any] = None,
    ):
        self.filename = filename
        self.r_cut = r_cut
        self.target_names = target_names
        self.log_target = log_target

        processed_dirname = f"processed_{'_'.join(target_names)}_rcut={self.r_cut}"

        # forward transform for targets
        if normalize_target:

            if normalizer_kwargs is None:
                normalizer_kwargs = {}

            # modify processed_dirname, since normalization will affect stored value
            kv_str = "_".join([f"{k}={v}" for k, v in normalizer_kwargs.items()])
            if kv_str:
                processed_dirname = processed_dirname + "_" + kv_str

            target_transform = ScalarTargetTransform(
                target_names=self.target_names,
                dataset_statistics_path=Path(root).joinpath(
                    processed_dirname, "dataset_statistics.pt"
                ),
                **normalizer_kwargs,
            )
        else:
            target_transform = None

        super().__init__(
            filenames=[filename],
            root=root,
            processed_dirname=processed_dirname,
            reuse=reuse,
            compute_dataset_statistics=compute_dataset_statistics,
            pre_transform=target_transform,
        )

    def get_data(self):
        filepath = self.raw_paths[0]
        df = pd.read_json(filepath, orient="split")

        assert "structure" in df.columns, (
            f"Unsupported task `{self.filename}`. Eigenn only works with data "
            "having geometric information (i.e. with `structure` in the matbench "
            "data). The provided dataset does not have this."
        )

        # convert structure
        df["structure"] = df["structure"].apply(lambda s: Structure.from_dict(s))

        # convert output to 2d tensor
        for name in self.target_names:
            df[name] = df[name].apply(lambda y: torch.atleast_2d(torch.as_tensor(y)))
            if self.log_target:
                df[name] = df[name].apply(lambda y: torch.log(y))

        crystals = []

        # convert to crystal data point
        for irow, row in df.iterrows():

            try:
                # get structure
                struct = row["structure"]

                # atomic numbers, shape (N_atom,)
                atomic_numbers = np.asarray(struct.atomic_numbers, dtype=np.int64)

                # get other property
                y = {name: row[name] for name in self.target_names}

                # other metadata needed by the model?

                c = Crystal.from_pymatgen(
                    struct=struct,
                    r_cut=self.r_cut,
                    x=None,
                    y=y,
                    atomic_numbers=atomic_numbers,
                )
                crystals.append(c)

            except Exception as e:
                warnings.warn(f"Failed converting structure {irow}, Skip it. {str(e)}")

        return crystals


class ScalarTargetTransform(torch.nn.Module):
    """
    Forward and inverse normalization.

    Forward is intended to be used as `pre_transform` of dataset and inverse is
    intended to be used before metrics and prediction function to transform the
    hessian back to the original space.

    Args:
        dataset_statistics_path: path to the dataset statistics file. Will be delayed to
            load when the forward or inverse function is called.
        target_names: names of the target

    """

    def __init__(
        self,
        target_names: List[str],
        dataset_statistics_path: Union[str, Path] = None,
    ):
        super().__init__()
        self.dataset_statistics_path = dataset_statistics_path
        self.dataset_statistics_loaded = False

        self.target_names = target_names

        self.normalizers = torch.nn.ModuleDict(
            {name: ScalarNormalize(num_features=1) for name in self.target_names}
        )

    def forward(self, struct: Crystal) -> Crystal:
        """
        Update target of Crystal.

        Note, should return it even we modify it in place.
        """

        # Instead of providing mean and norm of normalizer at instantiation, we delay
        # the loading here because this method will typically be called after dataset
        # statistics has been generated.
        self._fill_state_dict(struct.y[self.target_names[0]].device)

        for name in self.target_names:
            target = struct.y[name]
            struct.y[name] = self.normalizers[name](target)

        return struct

    def inverse(self, data: TensorType["batch", "D"], target_name: str):
        """
        Inverse transform model predictions/targets.

        This is supposed to be called in batch mode.
        """
        self._fill_state_dict(data.device)
        data = self.normalizers[target_name].inverse(data)

        return data

    def _fill_state_dict(self, device):
        """
        Because this is delayed be to called when actual forward or inverse is
        happening, need to move the tensor to the correct device.
        """
        if not self.dataset_statistics_loaded:
            statistics = torch.load(self.dataset_statistics_path)
            for name in self.target_names:
                self.normalizers[name].load_state_dict(statistics[name])
            self.to(device)

            self.dataset_statistics_loaded = True

    def compute_statistics(self, data: List[Crystal]) -> Dict[str, Any]:
        """
        Compute statistics of datasets.
        """

        targets = defaultdict(list)
        atomic_numbers = set()
        num_neigh = []

        for struct in data:
            for key in self.target_names:
                t = struct.y[key]
                targets[key].append(t)
            atomic_numbers.update(struct.atomic_numbers.tolist())
            num_neigh.append(struct.num_neigh)

        statistics = {
            "allowed_species": tuple(atomic_numbers),
            "average_num_neigh": torch.cat(num_neigh).mean(),
        }

        for key in self.target_names:
            t = torch.cat(targets[key])
            assert t.ndim == 2
            self.normalizers[key].compute_statistics(t)
            statistics[key] = self.normalizers[key].state_dict()

        return statistics


class StructureScalarDataModule(BaseDataModule):
    """
    Will search for fi`root/<trainset_filename>`.
    """

    def __init__(
        self,
        trainset_filename: str,
        valset_filename: str,
        testset_filename: str,
        target_names: List[str],
        *,
        r_cut: float,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        log_target: bool = True,
        compute_dataset_statistics: bool = True,
        normalize_target: bool = True,
        normalizer_kwargs: Dict[str, Any] = None,
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.r_cut = r_cut
        self.root = root
        self.target_names = target_names
        self.log_target = log_target
        self.compute_dataset_statistics = compute_dataset_statistics
        self.normalize_target = normalize_target
        self.normalizer_kwargs = normalizer_kwargs

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

        if self.compute_dataset_statistics:
            if self.normalizer_kwargs is None:
                kw = {}
            else:
                kw = self.normalizer_kwargs
            normalizer = ScalarTargetTransform(
                dataset_statistics_path=None, target_names=self.target_names, **kw
            )
            statistics_fn = normalizer.compute_statistics
        else:
            statistics_fn = None

        self.train_data = StructureScalarDataset(
            self.trainset_filename,
            self.r_cut,
            target_names=self.target_names,
            root=self.root,
            reuse=self.reuse,
            log_target=self.log_target,
            compute_dataset_statistics=statistics_fn,
            normalize_target=self.normalize_target,
            normalizer_kwargs=self.normalizer_kwargs,
        )
        self.val_data = StructureScalarDataset(
            self.valset_filename,
            self.r_cut,
            target_names=self.target_names,
            root=self.root,
            reuse=self.reuse,
            log_target=self.log_target,
            compute_dataset_statistics=None,
            normalize_target=self.normalize_target,
            normalizer_kwargs=self.normalizer_kwargs,
        )
        self.test_data = StructureScalarDataset(
            self.testset_filename,
            self.r_cut,
            target_names=self.target_names,
            root=self.root,
            reuse=self.reuse,
            log_target=self.log_target,
            compute_dataset_statistics=None,
            normalize_target=self.normalize_target,
            normalizer_kwargs=self.normalizer_kwargs,
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


if __name__ == "__main__":

    dm = StructureScalarDataModule(
        trainset_filename="crystal_elasticity_filtered_test.json",
        valset_filename="crystal_elasticity_filtered_test.json",
        testset_filename="crystal_elasticity_filtered_test.json",
        r_cut=5.0,
        target_names=["k_voigt", "k_reuss"],
        root="/Users/mjwen/Applications/eigenn_analysis/eigenn_analysis/dataset/elastic_tensor/20220523",
        reuse=False,
        normalize_target=True,
    )
    dm.prepare_data()
    dm.setup()
    dm.get_to_model_info()

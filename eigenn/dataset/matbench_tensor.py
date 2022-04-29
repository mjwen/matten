import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from e3nn.io import CartesianTensor
from pymatgen.core.structure import Structure
from torchtyping import TensorType

from eigenn.data.data import Crystal
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset
from eigenn.data.transform import MeanNormNormalize


class MatbenchTensorDataset(InMemoryDataset):
    """
    The matbench tensor (elastic, dielectric, piezoelastic) dataset of material
    properties.

    This will have whatever in the `column` as `y` for the dataset.

    Args:
        filename: matbench task filename, e.g. ``matbench_log_gvrh.json``. For a full
            list, see https://hackingmaterials.lbl.gov/automatminer/datasets.html
            Note, this should only be the filename, not the path name.
            If using local files, it should be in the `root` directory.
        r_cut: neighbor cutoff distance, in unit Angstrom.
        field_name: name of the cartesian tensor, e.g. `elastic_tensor_cartesian`
        root: root directory that stores the input and processed data.
        reuse: whether to reuse the preprocessed data.
        output_format: format of the target tensor, should be `cartesian` for `irreps`.
        output_formula: formula specifying symmetry of tensor. No matter what
            output_format is, output_formula should be given in cartesian notation.
            e.g. `ijkl=jikl=klij` for a elastic tensor.
          compute_dataset_statistics: callable to compute dataset statistics. Do not
            compute if `None`. Note, this is different from `normalize_target` below.
            This only determines whether to compute the statistics of the target of a
            dataset, and will generate a file named `dataset_statistics.pt` is $CWD
            if a callable is provided. Whether to use dataset statistics to do
            normalization is determined by `normalize_target`.
        normalize_target: whether to normalize the target hessian.
    """

    def __init__(
        self,
        filename: str,
        r_cut: float,
        field_name: str,
        *,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        output_format: str = "irreps",
        output_formula: str = "ijkl=jikl=klij",
        compute_dataset_statistics: Callable = None,
        normalize_target: bool = True,
        normalizer_kwargs: Dict[str, Any] = None,
    ):
        self.filename = filename
        self.r_cut = r_cut
        self.field_name = field_name
        self.output_format = output_format
        self.output_formula = output_formula

        processed_dirname = f"processed_rcut{self.r_cut}"

        if normalize_target and output_format == "cartesian":
            raise ValueError("Cannot normalize target for cartesian output")

        # forward transform for targets
        if normalize_target:
            if normalizer_kwargs is None:
                normalizer_kwargs = {}

            # modify processed_dirname, since normalization will affect stored value
            kv_str = "_".join([f"{k}-{v}" for k, v in normalizer_kwargs.items()])
            if kv_str:
                processed_dirname = processed_dirname + "_" + kv_str

            target_transform = TensorTargetTransform(
                Path(root).joinpath(processed_dirname, "dataset_statistics.pt"),
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

        # convert output to tensor
        df[self.field_name] = df[self.field_name].apply(lambda x: torch.as_tensor(x))

        # convert to irreps tensor is necessary
        if self.output_format == "irreps":
            converter = CartesianTensor(formula=self.output_formula)
            rtp = converter.reduced_tensor_products()
            df[self.field_name] = df[self.field_name].apply(
                lambda x: converter.from_cartesian(x, rtp).reshape(1, -1)
            )
        elif self.output_format == "cartesian":
            pass
        else:
            raise ValueError

        property_columns = [self.field_name]

        crystals = []

        # convert to crystal data point
        for irow, row in df.iterrows():

            try:
                # get structure
                struct = row["structure"]

                # atomic numbers, shape (N_atom,)
                atomic_numbers = np.asarray(struct.atomic_numbers, dtype=np.int64)

                # get other property
                y = {name: row[name] for name in property_columns}

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
                warnings.warn(f"Failed converting structure {irow}: {str(e)}. Skip it.")

        return crystals


def get_tensor_statistics(data: List[Crystal]) -> Dict[str, Any]:
    """
    Compute statistics of datasets.
    """

    elastic_tensors = []
    atomic_numbers = set()
    num_neigh = []

    key = "elastic_tensor_full"

    for struct in data:
        elastic_tensors.append(struct.y[key])
        atomic_numbers.update(struct.atomic_numbers.tolist())
        num_neigh.append(struct.num_neigh)

    elastic_tensors = torch.cat(elastic_tensors)
    atomic_numbers = tuple(atomic_numbers)
    average_num_neigh = torch.cat(num_neigh).mean()

    normalizer = MeanNormNormalize(irreps="2x0e+2x2e+4e")
    normalizer.compute_statistics(elastic_tensors)

    statistics = {
        key: normalizer.state_dict(),
        "allowed_species": atomic_numbers,
        "average_num_neigh": average_num_neigh,
    }

    return statistics


class TensorTargetTransform(torch.nn.Module):
    """
    Forward and inverse normalization of elastic tensor.

    Forward is intended to be used as `pre_transform` of dataset and inverse is
    intended to be used before metrics and prediction function to transform the
    hessian back to the original space.

    Args:
        dataset_statistics_path: path to the dataset statistics file. Will be delayed to
        load when the forward or inverse function is called.
    """

    def __init__(self, dataset_statistics_path: Union[str, Path], scale: float = 1.0):
        super().__init__()
        self.dataset_statistics_path = Path(dataset_statistics_path)
        self.dataset_statistics_loaded = False

        self.normalizer = MeanNormNormalize(irreps="2x0e+2x2e+4e", scale=scale)

    def forward(self, struct: Crystal) -> Crystal:
        """
        Update target of Crystal.

        Note, should return it even we modify it in place.
        """

        # Instead of providing mean and norm of normalizer at instantiation, we delay
        # the loading here because this method will typically be called after dataset
        # statistics has been generated.
        key = "elastic_tensor_full"

        self._fill_state_dict(struct.y[key].device)

        target = struct.y[key]  # shape (21,)
        struct.y[key] = self.normalizer(target)

        return struct

    def inverse(self, data: TensorType["batch", "D"]):
        """
        Inverse transform model predictions/targets.

        This is supposed to be called in batch mode.
        """
        self._fill_state_dict(data.device)
        data = self.normalizer.inverse(data)

        return data

    def _fill_state_dict(self, device):
        """
        Because this is delayed be to called when actual forward or inverse is
        happening, need to move the tensor to the correct device.
        """
        if not self.dataset_statistics_loaded:
            statistics = torch.load(self.dataset_statistics_path)
            self.normalizer.load_state_dict(statistics["elastic_tensor_full"])
            self.to(device)

            self.dataset_statistics_loaded = True


class MatbenchTensorDataModule(BaseDataModule):
    """
    Will search for fi`root/<trainset_filename>`.
    """

    def __init__(
        self,
        trainset_filename: str,
        valset_filename: str,
        testset_filename: str,
        field_name: str,
        *,
        r_cut: float,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        output_format: str = "cartesian",
        output_formula: str = "ijkl=jikl=klij",
        normalize_target: bool = True,
        normalizer_kwargs: Dict[str, Any] = None,
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        compute_dataset_statistics: bool = True,
        **kwargs,
    ):
        self.r_cut = r_cut
        self.root = root
        self.field_name = field_name
        self.output_format = output_format
        self.output_formula = output_formula
        self.normalize_target = normalize_target
        self.normalizer_kwargs = normalizer_kwargs

        self.compute_dataset_statistics = compute_dataset_statistics

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
            statistics_fn = get_tensor_statistics
        else:
            statistics_fn = None
        self.train_data = MatbenchTensorDataset(
            self.trainset_filename,
            self.r_cut,
            field_name=self.field_name,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
            compute_dataset_statistics=statistics_fn,
            normalize_target=self.normalize_target,
            normalizer_kwargs=self.normalizer_kwargs,
        )
        self.val_data = MatbenchTensorDataset(
            self.valset_filename,
            self.r_cut,
            field_name=self.field_name,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
            compute_dataset_statistics=None,
            normalize_target=self.normalize_target,
            normalizer_kwargs=self.normalizer_kwargs,
        )
        self.test_data = MatbenchTensorDataset(
            self.testset_filename,
            self.r_cut,
            field_name=self.field_name,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
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

    dm = MatbenchTensorDataModule(
        trainset_filename="crystal_elasticity_filtered_test.json",
        valset_filename="crystal_elasticity_filtered_test.json",
        testset_filename="crystal_elasticity_filtered_test.json",
        r_cut=5.0,
        field_name="elastic_tensor_full",
        output_format="irreps",
        output_formula="ijkl=jikl=klij",
        root="/Users/mjwen/Applications/eigenn_analysis/eigenn_analysis/dataset/elastic_tensor",
        reuse=False,
        normalize_target=True,
    )
    dm.prepare_data()
    dm.setup()
    dm.get_to_model_info()

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from e3nn.io import CartesianTensor
from pymatgen.core.structure import Structure

from eigenn.data.data import Crystal
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset
from eigenn.data.featurizer import GlobalFeaturizer
from eigenn.data.transform import FeatureTensorScalarTargetTransform


class TensorDataset(InMemoryDataset):
    """
    A dataset for tensors and derived properties, e.g. elastic tensor.

    This also deals with global features.
        - Note, the global features is dealt using the infrastructure to deal with
          scalar target.

    Args:
        filename: name of json data file.
        r_cut: neighbor cutoff distance, in unit Angstrom.
        tensor_target_name: name of the tensor in the dataset file..
        scalar_target_names: name of the derived scalar properties to be used as target.
        tensor_target_format: format of the target tensor: {`cartesian`, `irreps`}.
        tensor_target_formula: formula specifying symmetry of tensor. No matter what
            output_format is, output_formula should be given in cartesian notation.
            e.g. `ijkl=jikl=klij` for a elastic tensor.
        normalize_tensor_target: whether to normalize the tensor target
        log_scalar_targets: names of the scalar targets to be log transformed.
             Note, log is performed before computing any statistics, as a way to
             transform the target into a different space.
        normalize_scalar_targets: names of the scalar targets to be normalized.
        global_featurizer: featurizers to compute global features
        root: root directory that stores the input and processed data.
        reuse: whether to reuse the preprocessed data.
        compute_dataset_statistics: callable to compute dataset statistics. Do not
            compute if `None`. Note, this is different from `normalize_target` below.
            This only determines whether to compute the statistics of the target of a
            dataset, and will generate a file named `dataset_statistics.pt` is $CWD
            if a callable is provided. Whether to use dataset statistics to do
            normalization is determined by `normalize_target`.
    """

    def __init__(
        self,
        filename: str,
        r_cut: float,
        tensor_target_name: str,
        scalar_target_names: List[str],
        tensor_target_format: str = "irreps",
        tensor_target_formula: str = "ijkl=jikl=klij",
        normalize_tensor_target: bool = False,
        log_scalar_targets: List[str] = None,  # should always be None
        normalize_scalar_targets: List[str] = None,  # this should always be None
        global_featurizer: GlobalFeaturizer = None,
        *,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        compute_dataset_statistics: Callable = None,
    ):
        self.filename = filename
        self.tensor_target_name = tensor_target_name
        self.scalar_target_names = (
            [] if scalar_target_names is None else scalar_target_names
        )
        self.tensor_target_format = tensor_target_format
        self.tensor_target_formula = tensor_target_formula
        self.log_scalar_targets = log_scalar_targets
        self.r_cut = r_cut
        self.global_featurizer = global_featurizer

        if normalize_scalar_targets is not None:
            raise ValueError(
                "`normalize_scalar_targets` should always be `None` when tensor "
                "is trained, since we need to get the derived scalar properties from "
                "the tensor in its original space. Then the predicted scalar will "
                "always be in the unscaled space."
            )
        if log_scalar_targets is not None:
            raise ValueError(
                "`log_scalar_targets` should always be `None` when tensor "
                "is trained, since we need to get the derived scalar properties from "
                "the tensor in its original space. Then the predicted scalar will "
                "always be in the unscaled space."
            )

        processed_dirname = (
            f"processed_"
            f"tensor_name={tensor_target_name}."
            f"tensor_format={tensor_target_format}."
            f"normalize_tensor={normalize_tensor_target}."
            f"scalar_name={'-'.join(scalar_target_names)}."
            f"log_scalar={str(log_scalar_targets).replace(' ', '')}."
            f"global_featurizer={str(global_featurizer).replace(' ', '')}."
            f"normalize_scalar={str(normalize_scalar_targets).replace(' ', '')}"
        )

        # forward transform for targets
        if normalize_tensor_target or normalize_scalar_targets:
            if normalize_tensor_target:
                t_name = tensor_target_name
            else:
                t_name = None
            target_transform = FeatureTensorScalarTargetTransform(
                feature_names=["global_feats"],
                feature_sizes=[len(self.global_featurizer.feature_names)],
                tensor_target_name=t_name,
                scalar_target_names=normalize_scalar_targets,
                dataset_statistics_path="./dataset_statistics.pt",
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

        # convert tensor output to tensor
        df[self.tensor_target_name] = df[self.tensor_target_name].apply(
            lambda x: torch.as_tensor(x)
        )

        # get global features
        if self.global_featurizer is not None:
            df = self.global_featurizer(df)
            feats = df[self.global_featurizer.feature_names].to_numpy().tolist()
            df["global_feats"] = feats
        else:
            raise ValueError(f"Unsupported global_featurizer={self.global_featurizer}")

        # convert to irreps tensor is necessary
        if self.tensor_target_format == "irreps":
            converter = CartesianTensor(formula=self.tensor_target_formula)
            rtp = converter.reduced_tensor_products()
            df[self.tensor_target_name] = df[self.tensor_target_name].apply(
                lambda x: converter.from_cartesian(x, rtp).reshape(1, -1)
            )
        elif self.tensor_target_format == "cartesian":
            df[self.tensor_target_name] = df[self.tensor_target_name].apply(
                lambda x: torch.unsqueeze(x, 0)
            )
        else:
            raise ValueError(f"Unsupported oputput format")

        # convert scalar output to 2D shape
        for name in self.scalar_target_names:
            df[name] = df[name].apply(lambda y: torch.atleast_2d(torch.as_tensor(y)))

        # log scalar targets
        if self.log_scalar_targets is not None:
            for name in self.log_scalar_targets:
                df[name] = df[name].apply(lambda y: torch.log(y))

        target_columns = [self.tensor_target_name] + self.scalar_target_names

        crystals = []

        # convert to crystal data point
        for irow, row in df.iterrows():

            try:
                # get structure
                struct = row["structure"]

                # atomic numbers, shape (N_atom,)
                atomic_numbers = np.asarray(struct.atomic_numbers, dtype=np.int64)

                # get targets
                y = {name: row[name] for name in target_columns}

                # feats
                x = {
                    "global_feats": torch.reshape(  # reshape to be a 2D tensor
                        torch.as_tensor(row["global_feats"]), (1, -1)
                    )
                }

                # other metadata needed by the model?

                c = Crystal.from_pymatgen(
                    struct=struct,
                    r_cut=self.r_cut,
                    x=x,
                    y=y,
                    atomic_numbers=atomic_numbers,
                )
                crystals.append(c)

            except Exception as e:
                warnings.warn(f"Failed converting structure {irow}, Skip it. {str(e)}")

        return crystals


class TensorDataModule(BaseDataModule):
    def __init__(
        self,
        trainset_filename: str,
        valset_filename: str,
        testset_filename: str,
        *,
        r_cut: float,
        tensor_target_name: str,
        scalar_target_names: List[str] = None,
        tensor_target_format: str = "irreps",
        tensor_target_formula: str = "ijkl=jikl=klij",
        normalize_tensor_target: bool = False,
        log_scalar_targets: List[str] = None,
        normalize_scalar_targets: List[str] = None,
        global_featurizer: str = "default",
        root: Union[str, Path] = ".",
        reuse: bool = True,
        compute_dataset_statistics: bool = True,
        loader_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.tensor_target_name = tensor_target_name
        self.tensor_target_formula = tensor_target_formula
        self.tensor_target_format = tensor_target_format
        self.normalize_tensor_target = normalize_tensor_target
        self.scalar_target_names = (
            [] if scalar_target_names is None else scalar_target_names
        )
        self.log_scalar_targets = log_scalar_targets
        self.normalize_scalar_targets = normalize_scalar_targets
        self.global_featurizer = global_featurizer
        self.compute_dataset_statistics = compute_dataset_statistics
        self.r_cut = r_cut
        self.root = root

        super().__init__(
            trainset_filename,
            valset_filename,
            testset_filename,
            reuse=reuse,
            loader_kwargs=loader_kwargs,
        )

    def setup(self, stage: Optional[str] = None):

        if self.compute_dataset_statistics:

            # get global features names
            if self.global_featurizer == "default":
                gf = GlobalFeaturizer()
                gf_name = "global_feats"
                gf_size = len(gf.feature_names)
            else:
                raise ValueError(
                    f"Unsupported global_featurizer={self.global_featurizer}"
                )

            # NOTE, abuse scalar_target_names -- using it to normalize features as well
            normalizer = FeatureTensorScalarTargetTransform(
                feature_names=[gf_name],
                feature_sizes=[gf_size],
                tensor_target_name=self.tensor_target_name,
                scalar_target_names=self.scalar_target_names,
                dataset_statistics_path=None,
            )
            statistics_fn = normalizer.compute_statistics
        else:
            statistics_fn = None

        self.train_data = TensorDataset(
            self.trainset_filename,
            r_cut=self.r_cut,
            tensor_target_name=self.tensor_target_name,
            tensor_target_format=self.tensor_target_format,
            tensor_target_formula=self.tensor_target_formula,
            normalize_tensor_target=self.normalize_tensor_target,
            scalar_target_names=self.scalar_target_names,
            log_scalar_targets=self.log_scalar_targets,
            normalize_scalar_targets=self.normalize_scalar_targets,
            global_featurizer=gf,
            root=self.root,
            reuse=self.reuse,
            compute_dataset_statistics=statistics_fn,
        )

        self.val_data = TensorDataset(
            self.valset_filename,
            r_cut=self.r_cut,
            tensor_target_name=self.tensor_target_name,
            tensor_target_format=self.tensor_target_format,
            tensor_target_formula=self.tensor_target_formula,
            normalize_tensor_target=self.normalize_tensor_target,
            scalar_target_names=self.scalar_target_names,
            log_scalar_targets=self.log_scalar_targets,
            normalize_scalar_targets=self.normalize_scalar_targets,
            global_featurizer=gf,
            root=self.root,
            reuse=self.reuse,
            compute_dataset_statistics=None,
        )

        self.test_data = TensorDataset(
            self.testset_filename,
            r_cut=self.r_cut,
            tensor_target_name=self.tensor_target_name,
            tensor_target_format=self.tensor_target_format,
            tensor_target_formula=self.tensor_target_formula,
            normalize_tensor_target=self.normalize_tensor_target,
            scalar_target_names=self.scalar_target_names,
            log_scalar_targets=self.log_scalar_targets,
            normalize_scalar_targets=self.normalize_scalar_targets,
            global_featurizer=gf,
            root=self.root,
            reuse=self.reuse,
            compute_dataset_statistics=None,
        )

    # TODO this needs to be removed
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

    dm = TensorDataModule(
        trainset_filename="crystal_elasticity_n20.json",
        valset_filename="crystal_elasticity_n20.json",
        testset_filename="crystal_elasticity_n20.json",
        r_cut=5.0,
        tensor_target_name="elastic_tensor_full",
        scalar_target_names=["k_voigt", "k_reuss"],
        root="/Users/mjwen.admin/Packages/eigenn_analysis/eigenn_analysis/dataset"
        "/elastic_tensor/20220714",
        reuse=False,
    )

    dm.prepare_data()
    dm.setup()
    dm.get_to_model_info()

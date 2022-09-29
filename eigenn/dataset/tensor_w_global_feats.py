import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from e3nn.io import CartesianTensor
from monty.serialization import loadfn
from pymatgen.core.structure import Structure

from eigenn.data.data import Crystal
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset
from eigenn.data.featurizer import GlobalFeaturizer
from eigenn.data.transform import FeatureTensorScalarTargetTransform


class TensorDataset(InMemoryDataset):
    """
    A dataset for tensors (e.g. elastic) and derived scalar properties (e.g. bulk
    modulus).

    This also provide the possibility to use global features.

    Args:
        filename: name of input data file.
        r_cut: neighbor cutoff distance, in unit Angstrom.
        tensor_target_name: name of the tensor target.
        tensor_target_format: {`cartesian`, `irreps`}. The target tensor in the
            data file is provided in cartesian or irreps format.
        tensor_target_formula: formula specifying symmetry of tensor, e.g.
            `ijkl=jikl=klij` for a elastic tensor.
        normalize_tensor_target: whether to normalize the tensor target.
        scalar_target_names: names of the target scalar properties.
        log_scalar_targets: whether to log transform the scalar targets; one for each
            scalar target given in `scalar_target_names`. Note, log is performed before
            computing any statistics, controlled by `normalize_scalar_targets`.
            If `None`, no log is performed.
        normalize_scalar_targets: whether to normalize the scalar targets; one for each
            scalar target given in `scalar_target_names`. If `None`, no normalize is
            performed.
        global_featurizer: featurizer to compute global features.
        normalize_global_feature: whether to normalize the global feature.
        root: root directory that stores the input and processed data.
        reuse: whether to reuse the preprocessed data.
        dataset_statistics_fn: callable to compute dataset statistics. If `None`, do not
            compute statistics. Note this is typically used together with
            `normalize_tensor_target`, `normalize_scalar_targets`,
            and `normalize_global_feature`. However, as long as `dataset_statistics_fn`
            is provided, the statistics will be computed, and a file named
            `dataset_statistics.pt` is generated in $CWD. Whether to use the computed
            dataset statistics for normalization is determined by the normalize flags.
    """

    def __init__(
        self,
        filename: str,
        r_cut: float,
        *,
        tensor_target_name: str = None,
        tensor_target_format: str = "irreps",
        tensor_target_formula: str = "ijkl=jikl=klij",
        normalize_tensor_target: bool = False,
        scalar_target_names: List[str] = None,
        log_scalar_targets: List[bool] = None,
        normalize_scalar_targets: List[bool] = None,
        global_featurizer: GlobalFeaturizer = None,
        normalize_global_feature: bool = False,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        dataset_statistics_fn: Callable = None,
    ):
        self.filename = filename
        self.r_cut = r_cut

        self.tensor_target_name = tensor_target_name
        self.tensor_target_format = tensor_target_format
        self.tensor_target_formula = tensor_target_formula
        self.normalize_tensor_target = normalize_tensor_target

        self.scalar_target_names = (
            [] if scalar_target_names is None else scalar_target_names
        )
        self.log_scalar_targets = (
            [False] * len(self.scalar_target_names)
            if log_scalar_targets is None
            else log_scalar_targets
        )
        self.normalize_scalar_targets = (
            [False] * len(self.scalar_target_names)
            if normalize_scalar_targets is None
            else normalize_scalar_targets
        )

        self.global_featurizer = global_featurizer
        self.normalize_global_feature = normalize_global_feature

        # check compatibility
        # if normalize_scalar_targets is not None:
        #     raise ValueError(
        #         "`normalize_scalar_targets` should always be `None` when tensor "
        #         "is trained, since we need to get the derived scalar properties from "
        #         "the tensor in its original space. Then the predicted scalar will "
        #         "always be in the unscaled space."
        #     )
        # if log_scalar_targets is not None:
        #     raise ValueError(
        #         "`log_scalar_targets` should always be `None` when tensor "
        #         "is trained, since we need to get the derived scalar properties from "
        #         "the tensor in its original space. Then the predicted scalar will "
        #         "always be in the unscaled space."
        #     )

        if self.normalize_global_feature and self.global_featurizer is None:
            raise ValueError(
                "`normalize_global_feature=True`, but `global_featurizer=None`"
            )

        processed_dirname = (
            f"processed_"
            f"tensor_name={self.tensor_target_name}-"
            f"tensor_format={self.tensor_target_format}-"
            f"normalize_tensor={self.normalize_tensor_target}-"
            f"scalar_names={'-'.join(self.scalar_target_names)}_"
            f"log_scalars={str(self.log_scalar_targets).replace(' ', '')}-"
            f"normalize_scalars={str(self.normalize_scalar_targets).replace(' ', '')}-"
            f"normalize_global_feat={self.normalize_global_feature}"
        )

        # Normalize tensor/scalar targets and global features
        if (
            self.normalize_tensor_target
            or any(self.normalize_scalar_targets)
            or self.normalize_global_feature
        ):
            if normalize_tensor_target:
                t_name = tensor_target_name
            else:
                t_name = None

            s_names = [
                s
                for s, n in zip(self.scalar_target_names, self.normalize_scalar_targets)
                if n
            ]
            if not s_names:
                s_names = None

            if self.normalize_global_feature:
                f_names = ["global_feats"]
                f_sizes = [len(self.global_featurizer.feature_names)]
            else:
                f_names = None
                f_sizes = None

            pre_transform = FeatureTensorScalarTargetTransform(
                feature_names=f_names,
                feature_sizes=f_sizes,
                tensor_target_name=t_name,
                scalar_target_names=s_names,
                dataset_statistics_path="./dataset_statistics.pt",
            )
        else:
            pre_transform = None

        super().__init__(
            filenames=[filename],
            root=root,
            processed_dirname=processed_dirname,
            reuse=reuse,
            dataset_statistics_fn=dataset_statistics_fn,
            pre_transform=pre_transform,
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
                gf = torch.as_tensor(row["global_feats"])
                if torch.isnan(gf).any():
                    raise ValueError("NaN in global feats")
                x = {
                    "global_feats": torch.reshape(gf, (1, -1))  # reshape to a 2D tensor
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
        global_featurizer: str = None,
        normalize_global_features: bool = False,
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
        self.normalize_global_features = normalize_global_features
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

        # global featurizer
        if self.global_featurizer.endswith(".yaml"):
            feature_names = loadfn(self.global_featurizer)
            gf = GlobalFeaturizer(feature_names=feature_names)
            gf_name = "global_feats"
            gf_size = len(gf.feature_names)
        else:
            raise ValueError(f"Unsupported global_featurizer={self.global_featurizer}")

        if self.compute_dataset_statistics:

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
            normalize_global_feature=self.normalize_global_features,
            root=self.root,
            reuse=self.reuse,
            dataset_statistics_fn=statistics_fn,
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
            normalize_global_feature=self.normalize_global_features,
            root=self.root,
            reuse=self.reuse,
            dataset_statistics_fn=None,
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
            normalize_global_feature=self.normalize_global_features,
            root=self.root,
            reuse=self.reuse,
            dataset_statistics_fn=None,
        )

    # TODO this needs to be removed
    def get_to_model_info(self) -> Dict[str, Any]:
        atomic_numbers = set()
        num_neigh = []
        for data in self.train_dataloader():
            a = data.atomic_numbers.tolist()
            atomic_numbers.update(a)
            num_neigh.append(data.num_neigh)

        global_feats_size = data.x["global_feats"].shape[1]

        # .item to convert to float so that lightning cli can save it to yaml
        average_num_neighbors = torch.mean(torch.cat(num_neigh)).item()

        return {
            "allowed_species": tuple(atomic_numbers),
            "average_num_neighbors": average_num_neighbors,
            "global_feats_size": global_feats_size,
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

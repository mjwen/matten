import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from e3nn.io import CartesianTensor
from pymatgen.core.structure import Structure

from eigenn.data.data import Crystal
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset


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
    """

    def __init__(
        self,
        filename: str,
        r_cut: float,
        field_name: str,
        *,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        output_format: str = "cartesian",
        output_formula: str = "ijkl=jikl=klij",
    ):
        self.filename = filename
        self.r_cut = r_cut
        self.field_name = field_name
        self.output_format = output_format
        self.output_formula = output_formula

        super().__init__(
            filenames=[filename],
            root=root,
            processed_dirname=f"processed_rcut{self.r_cut}",
            reuse=reuse,
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
        output = df[self.field_name].apply(lambda x: torch.as_tensor(x))

        # convert to irreps tensor is necessary
        converter = CartesianTensor(formula=self.output_formula)
        if self.output_format == "irreps":
            output = converter.from_cartesian(output)
        df[self.field_name] = output

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


class MatbenchTensorDataMoldule(BaseDataModule):
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
        output_format: str = "cartesian",
        output_formula: str = "ijkl=jikl=klij",
        reuse: bool = True,
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.r_cut = r_cut
        self.root = root
        self.field_name = field_name
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
        self.train_data = MatbenchTensorDataset(
            self.trainset_filename,
            self.r_cut,
            field_name=self.field_name,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
        )
        self.val_data = MatbenchTensorDataset(
            self.valset_filename,
            self.r_cut,
            field_name=self.field_name,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
        )
        self.test_data = MatbenchTensorDataset(
            self.testset_filename,
            self.r_cut,
            field_name=self.field_name,
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


if __name__ == "__main__":

    dm = MatbenchDataMoldule(
        trainset_filename="/Users/mjwen/Applications/eigenn_analysis/eigenn_analysis/dataset/elastic_tensor/crystal_elasticity.json",
        valset_filename="/Users/mjwen/Applications/eigenn_analysis/eigenn_analysis/dataset/elastic_tensor/crystal_elasticity.json",
        testset_filename="/Users/mjwen/Applications/eigenn_analysis/eigenn_analysis/dataset/elastic_tensor/crystal_elasticity.json",
        r_cut=5.0,
        field_name="elastic_tensor_cartesian",
        output_format="cartesian",
        output_formula="ijkl=jikl=klij",
        root="/tmp",
        reuse=False,
    )
    dm.prepare_data()
    dm.setup()
    dm.get_to_model_info()

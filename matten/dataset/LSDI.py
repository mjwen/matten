import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from e3nn.io import CartesianTensor
from monty.serialization import loadfn

from matten.data.data import Crystal
from matten.data.datamodule import BaseDataModule
from matten.data.dataset import InMemoryDataset


class SiNMRDataset(InMemoryDataset):
    """
    The LSDI dataset of 29Si NMR nuclear shielding tensors.

    Args:
        filename: 29Si NMR tensor filename, e.g. ``LSDI_NMR_tensors.json``. Note, this
            should only be the filename, not the full path. This file will be
            searched in `root`.
        r_cut: neighbor cutoff distance, in unit Angstrom.
        root: root directory that stores the input and processed data.
        reuse: whether to reuse the preprocessed data.
        unpack: If True, each structure will have a single shielding tensor (structures
            will be repeated if they contain multiple shielding tensors). If False,
            the structure will have as many shielding tensors as unique sites.
        output_format: format of the target tensor, should be `cartesian` for `irreps`.
        output_formula: formula specifying symmetry of tensor. No matter what
            output_format is, output_formula should be given in cartesian notation.
            e.g. `ij=ji` for a general 2D tensor and `ij=ji` for a symmetric 2D tensor.
    """

    def __init__(
        self,
        filename: str,
        r_cut: float,
        *,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        unpack: bool = True,
        output_format: str = "cartesian",
        output_formula: str = "ij=ji",
    ):
        self.filename = filename
        self.r_cut = r_cut
        self.unpack = unpack

        assert output_format in ("cartesian", "irreps")
        self.output_format = output_format
        self.output_formula = output_formula

        super().__init__(
            filenames=[filename],
            processed_dirname=f"processed_rcut-{self.r_cut}_format-{self.output_format}",
            root=root,
            reuse=reuse,
        )

    def get_data(self):
        filepath = self.raw_paths[0]
        data = loadfn(filepath)

        # unpacks data as described above
        if self.unpack:
            unpacked_data = []
            for entry in data:
                for idx in range(len(entry["tensor"])):
                    node_index = entry["ind"][idx]
                    node_mask = np.zeros(len(entry["structure"].sites), int)
                    node_mask[node_index] = 1
                    d = {
                        "structure": entry["structure"],
                        "tensor": [entry["tensor"][idx]],  # shape 1x3x3
                        "index": node_mask,
                    }
                    unpacked_data.append(d)
            data = unpacked_data

        crystals = []

        # converter for irreps tensor
        converter = CartesianTensor(formula=self.output_formula)

        # convert to crystal data point
        for irow, row in enumerate(data):

            output = torch.as_tensor(row["tensor"])
            if self.output_format == "irreps":
                output = converter.from_cartesian(output)

            try:
                # get structure
                struct = row["structure"]

                # get property
                # y = {'tensor_output': value,
                #     'node_masks': [True or False]
                #     }
                # where value is nx3x3
                y = {
                    "tensor_output": output,
                    "node_masks": np.asarray(row["index"], dtype=bool),
                }
                # atomic numbers, shape (N_atom,)
                atomic_numbers = np.asarray(struct.atomic_numbers, dtype=np.int64)

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


class SiNMRDataModule(BaseDataModule):
    """
    Will search for files at, e.g. `root/<trainset_filename>`.
    """

    def __init__(
        self,
        trainset_filename: str,
        valset_filename: str,
        testset_filename: str,
        *,
        r_cut: float,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        output_format: str = "cartesian",
        output_formula: str = "ij=ji",
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.r_cut = r_cut
        self.root = root
        self.reuse = reuse
        self.output_format = output_format
        self.output_formula = output_formula

        super().__init__(
            trainset_filename,
            valset_filename,
            testset_filename,
            state_dict_filename=state_dict_filename,
            restore_state_dict_filename=restore_state_dict_filename,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        self.train_data = SiNMRDataset(
            self.trainset_filename,
            self.r_cut,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
        )
        self.val_data = SiNMRDataset(
            self.valset_filename,
            self.r_cut,
            root=self.root,
            reuse=self.reuse,
            output_format=self.output_format,
            output_formula=self.output_formula,
        )
        self.test_data = SiNMRDataset(
            self.testset_filename,
            self.r_cut,
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

    dm = SiNMRDataModule(
        trainset_filename="../../datasets/LSDI_NMR_tensor.json",
        valset_filename="../../datasets/LSDI_NMR_tensor.json",
        testset_filename="../../datasets/LSDI_NMR_tensor.json",
        r_cut=5.0,
        reuse=False,
        root=Path.cwd().joinpath("LSDI_NMR"),
    )
    dm.prepare_data()
    dm.setup()
    dm.get_to_model_info()

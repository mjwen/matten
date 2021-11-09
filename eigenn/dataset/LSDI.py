import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from monty.serialization import loadfn

from eigenn.data.data import Crystal
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset


class SiNMRDataset(InMemoryDataset):
    """
    The LSDI dataset of 29Si NMR nuclear shielding tensors.

    Args:
        filename: 29Si NMR tensor filename, e.g. ``LSDI_NMR_tensors.json``.
            Note, this should only be the filename, not the path name.
            If using local files, it should be placed on a path with `raw` before the
            filename. For example, <root>/raw/Si_tasks.json
            where <root> is the path provided by `root`.
        r_cut: neighbor cutoff distance, in unit Angstrom.
        root: root directory of the data, will contain `raw` and `processed` files.
        unpack: If True, each structure will have a single shielding tensor (structures
            will be repeated if they contain multiple shielding tensors). If False,
            the structure will have as many shielding tensors as unique sites.
    """

    def __init__(self, filename: str, r_cut: float, root=".", unpack: bool = True):
        self.r_cut = r_cut
        self.filename = filename
        self.unpack = unpack
        super().__init__(filenames=[filename], root=root)

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
        # convert to crystal data point
        for irow, row in enumerate(data):

            try:
                # get structure
                struct = row["structure"]

                # get property
                # y = {'tensor_output': value,
                #     'node_masks': [True or False]
                #     }
                # where value is nx3x3
                y = {
                    "tensor_output": np.asarray(row["tensor"]),
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


class SiNMRDataMoldule(BaseDataModule):
    """
    Will search for files at, e.g. `root/raw/<trainset_filename>`.
    """

    def __init__(
        self,
        trainset_filename: Union[str, Path],
        valset_filename: Union[str, Path],
        testset_filename: Union[str, Path],
        *,
        r_cut: float,
        root: Union[str, Path] = ".",
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self.r_cut = r_cut
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
        self.train_data = SiNMRDataset(self.trainset_filename, self.r_cut, self.root)
        self.val_data = SiNMRDataset(self.valset_filename, self.r_cut, self.root)
        self.test_data = SiNMRDataset(self.testset_filename, self.r_cut, self.root)

    def get_to_model_info(self) -> Dict[str, Any]:

        # TODO This Should be moved to dataset
        atomic_numbers = set()
        for data in self.train_dataloader():
            a = data.atomic_numbers.tolist()
            atomic_numbers.update(a)
        num_species = len(atomic_numbers)

        # return {"num_species": num_species}

        return {"allowed_species": tuple(atomic_numbers)}


if __name__ == "__main__":

    dm = SiNMRDataMoldule(
        trainset_filename="LSDI_NMR_tensor.json",
        valset_filename="LSDI_NMR_tensor.json",
        testset_filename="LSDI_NMR_tensor.json",
        r_cut=5.0,
        root=Path.cwd().joinpath("LSDI_NMR"),
    )
    dm.prepare_data()
    dm.setup()
    dm.get_to_model_info()

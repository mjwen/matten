import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from monty.serialization import loadfn

from eigenn.data.data import Crystal
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset


# TODO, keep info of cut. If cutoff changes, should reprocess data.
class MatbenchDataset(InMemoryDataset):
    """
    The matbench dataset of material properties.

    This will have whatever in the `column` as `y` for the dataset.

    Args:
        filename: matbench task filename, e.g. ``matbench_log_gvrh.json``. For a full
            list, see https://hackingmaterials.lbl.gov/automatminer/datasets.html
            Note, this should only be the filename, not the path name.
            If using local files, it should be in the `root` directory.
        r_cut: neighbor cutoff distance, in unit Angstrom.
        root: root directory that stores the input and processed data.
    """

    MATBENCH_WEBSITE = "https://hackingmaterials.lbl.gov/automatminer/datasets.html"

    # tasks with structure info and their target
    WITH_STRUCTURE = {
        "matbench_dielectric": "n",
        "matbench_jdft2d": "exfoliation_en",
        "matbench_log_gvrh": "log10(G_VRH)",
        "matbench_log_kvrh": "log10(K_VRH)",
        "matbench_mp_e_form": "e_form",
        "matbench_mp_gap": "gap pbe",
        "matbench_mp_is_metal": "is_metal",
        "matbench_perovskites": "e_form",
        "matbench_phonons": "last phdos peak",
    }

    def __init__(self, filename: str, r_cut: float, root: Union[str, Path] = "."):
        self.filename = filename
        self.r_cut = r_cut

        url = f"https://ml.materialsproject.org/projects/{filename}.gz"
        super().__init__(
            filenames=[filename],
            root=root,
            processed_dirname=f"processed_rcut{self.r_cut}",
            url=url,
        )

    @classmethod
    def from_task_name(cls, task_name: str, r_cut: float, root="."):
        filename = f"{task_name}.json"
        return cls(filename=filename, r_cut=r_cut, root=root)

    def get_data(self):
        filepath = self.raw_paths[0]
        data = loadfn(filepath)

        columns = data["columns"]
        try:
            struct_idx = columns.index("structure")
        except ValueError:
            raise ValueError(
                f"Unsupported task `{self.filename}`. Eigenn only works with data "
                "having geometric information (i.e. with `structure` in the matbench "
                "data). The provided dataset does not have this. Matbench tasks with "
                f"`structure` information include: "
                f"{', '.join(self.WITH_STRUCTURE.keys())}. "
                f"See {self.MATBENCH_WEBSITE} for more."
            )

        crystals = []

        # convert to crystal data point
        for irow, row in enumerate(data["data"]):

            try:
                # get structure
                struct = row[struct_idx]

                # get property
                y = {}
                for i, (name, value) in enumerate(zip(columns, row)):
                    # not structure column, then property column
                    if i != struct_idx:
                        y[name] = np.atleast_1d(value)

                # metadata needed by the model

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
                raise Exception

        return crystals


class MatbenchDataMoldule(BaseDataModule):
    """
    Will search for files at, e.g. `root/<trainset_filename>`.
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
        self.train_data = MatbenchDataset(self.trainset_filename, self.r_cut, self.root)
        self.val_data = MatbenchDataset(self.valset_filename, self.r_cut, self.root)
        self.test_data = MatbenchDataset(self.testset_filename, self.r_cut, self.root)

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
        trainset_filename="matbench_dielectric.json",
        valset_filename="matbench_dielectric.json",
        testset_filename="matbench_dielectric.json",
        r_cut=5.0,
        root="/tmp",
    )
    dm.prepare_data()
    dm.setup()
    dm.get_to_model_info()

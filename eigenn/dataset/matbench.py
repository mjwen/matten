from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from monty.serialization import loadfn

from eigenn.data.data import Crystal
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset


class MatbenchDataset(InMemoryDataset):
    """
    The matbench dataset of material properties.

    This will have whatever in the `column` as `y` for the dataset.

    Args:
        task_name: matbench task names, e.g. ``matbench_log_gvrh``. For a fully list, see
            https://hackingmaterials.lbl.gov/automatminer/datasets.html
        r_cut: neighbor cutoff distance, in unit Angstrom.
        root:
    """

    def __init__(self, filename, r_cut: float, root="."):
        self.r_cut = r_cut

        url = f"https://ml.materialsproject.org/projects/{filename}.gz"
        super().__init__(filenames=[filename], root=root, url=url)

    @classmethod
    def from_task_name(cls, task_name: str, r_cut: float, root="."):
        filename = f"{task_name}.json"
        return cls(filename=filename, r_cut=r_cut, root=root)

    def get_data(self):
        filepath = self.raw_paths[0]
        data = loadfn(filepath)

        columns = data["columns"]
        struct_idx = columns.index("structure")

        crystals = []

        # convert to crystal data point
        for irow, row in enumerate(data["data"]):

            try:
                # get structure
                struct = row[struct_idx]

                # get property
                y = {
                    name: value
                    for i, (name, value) in enumerate(zip(columns, row))
                    if i != struct_idx
                }

                # metadata needed by the model

                # atomic numbers, shape (N_atom, 1)
                atomic_numbers = np.asarray(
                    struct.atomic_numbers, dtype=np.int64
                ).reshape(-1, 1)

                c = Crystal.from_pymatgen(
                    struct=struct,
                    r_cut=self.r_cut,
                    x=None,
                    y=y,
                    atomic_numbers=atomic_numbers,
                )
                crystals.append(c)

            except Exception as e:
                raise Exception(f"Failed converting structure {irow}. " + str(e))

        return crystals


class MatbenchDataMoldule(BaseDataModule):
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
        self.train_data = MatbenchDataset(self.trainset_filename, self.r_cut, self.root)
        self.val_data = MatbenchDataset(self.valset_filename, self.r_cut, self.root)
        self.test_data = MatbenchDataset(self.testset_filename, self.r_cut, self.root)

    def get_to_model_info(self) -> Dict[str, Any]:

        # TODO This Should be moved to dataset
        atomic_numbers = set()
        for data in self.train_dataloader():
            a = data.atomic_numbers[0].reshape(-1).tolist()
            atomic_numbers.update(a)
        num_species = len(atomic_numbers)

        return {"num_species": num_species}


if __name__ == "__main__":
    dataset = MatbenchDataset.from_task_name(
        task_name="matbench_dielectric", r_cut=5.0, root="/tmp"
    )

    dm = MatbenchDataMoldule(
        trainset_filename="matbench_dielectric.json",
        valset_filename="matbench_dielectric.json",
        testset_filename="matbench_dielectric.json",
        r_cut=5.0,
        root="/tmp",
    )
    dm.setup()
    dm.get_to_model_info()

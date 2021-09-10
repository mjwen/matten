from monty.serialization import loadfn

from eigenn.data.data import Crystal
from eigenn.data.dataset import InMemoryDataset


class MatbenchDataset(InMemoryDataset):
    """
    Args:
        task_name: matbench task names, e.g. ``matbench_log_gvrh``. For a fully list, see
            https://hackingmaterials.lbl.gov/automatminer/datasets.html
        r_cut: neighbor cutoff distance, in unit Angstrom.
        root:
    """

    def __init__(self, task_name: str, r_cut: float, root="."):
        self.task_name = task_name
        self.r_cut = r_cut

        filename = f"{task_name}.json"
        url = f"https://ml.materialsproject.org/projects/{filename}.gz"

        super().__init__(filenames=[filename], url=url, root=root)

    def get_data(self):
        filepath = self.raw_paths[0]
        data = loadfn(filepath)

        columns = data["columns"]
        struct_idx = columns.index("structure")

        crystals = []

        # convert to crystal data point
        for irow, row in enumerate(data["data"][:10]):

            try:
                # get structure
                struct = row[struct_idx]

                # get property
                y = {
                    name: value
                    for i, (name, value) in enumerate(zip(columns, row))
                    if i != struct_idx
                }

                c = Crystal.from_pymatgen(struct=struct, r_cut=self.r_cut, x=y, y=y)
                crystals.append(c)
            except Exception as e:
                raise Exception(f"Failed converting structure {irow}. " + str(e))

        return crystals


if __name__ == "__main__":
    dataset = MatbenchDataset(task_name="matbench_dielectric", r_cut=5.0)

# -*- coding: utf-8 -*-
import warnings

import numpy as np
import torch
from monty.serialization import loadfn

from eigenn.data.data import Crystal
from eigenn.data.datamodule import BaseDataModule
from eigenn.data.dataset import InMemoryDataset


class SiNMRDataset(InMemoryDataset):
    """
    The LSDI dataset of 29Si NMR nuclear shielding tensors.

    Args:
        filename: 29Si NMR tensor filename, e.g. ``Si_tasks.json``.
            Note, this should only to be filename, not the path name.
            If using local files, it should be placed on a path with `raw` before the
            filename. For example, <root>/raw/Si_tasks.json
            where <root> is the path provided by `root`.

        r_cut: neighbor cutoff distance, in unit Angstrom.
        root: root directory of the data, will contain `raw` and `processed` files.
        unpack: If True, each structure will have a single shielding tensor (structures
            will be repeated if they contain multiple shielding tensors). If False,
            the structure will have as many shielding tensors as unique sites.


    """

    def __init__(self, filename, r_cut: float, root=".", unpack=True):
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
                        "tensor": entry["tensor"][idx],
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
                # y = {'tensor_output': value, 'node_masks': torch.tensor([True or False])}
                # where value is nx3x3
                y = {
                    "tensor_output": row["tensor"],
                    "node_masks": torch.as_tensor(row["index"], dtype=torch.bool),
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

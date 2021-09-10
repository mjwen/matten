from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
import torch
import tqdm
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset as PyGInMemoryDataset
from torch_geometric.data import extract_bz2, extract_gz, extract_tar, extract_zip

from eigenn.data.data import DataPoint
from eigenn.utils import to_list, to_path


class InMemoryDataset(PyGInMemoryDataset):
    """
    Base in memory dataset.

    Subclass must implement:
      - get_data()

    Args:
        filenames: filenames of the dataset. Will try to find the files with `filenames` in
            `root/raw/`. If they exist, will use them. If not, will try to download
            from `url`. This should not be a path, but just the names.
        root: root to store the dataset.
        url: path to download the dataset.
    """

    def __init__(
        self,
        filenames: List[Union[str, Path]],
        root: Union[str, Path] = ".",
        url: Optional[str] = None,
    ):
        self.filenames = to_list(filenames)
        self.url = url

        self.data = None
        self.slices = None

        # !!! don't delete this block.
        # otherwise the inherent children class will ignore the download function here
        class_type = type(self)
        if class_type != InMemoryDataset:
            if "download" not in self.__class__.__dict__:
                class_type.download = InMemoryDataset.download
            if "process" not in self.__class__.__dict__:
                class_type.process = InMemoryDataset.process

        super().__init__(root=root)

        if self.data is None or self.slices is None:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get_data(self) -> List[DataPoint]:
        """
        Process the (downloaded) files to get a list of data points.

        In this function, the files in ``self.raw_file_names`` (this is basically
        `<root>/raw/<filenames>` with `root` and `filenames` provided at the
        instantiation of the class) should be processed to generate a list of
        ``DataPoint`` object.
        """
        raise NotImplementedError

    @property
    def raw_file_names(self):
        return self.filenames

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # from torch_geometric.data import download_url
        # download_url(self.url, self.raw_dir)
        ## download_url does not work for some url, e.g. matbench ones

        filepath = _fetch_external_dataset(self.url, self.raw_dir)
        _extract_file(filepath, self.raw_dir)

    def process(self):
        data_list = self.get_data()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        self.data = data
        self.slices = slices

        torch.save((data, slices), self.processed_paths[0])

    # This is copied from  InMemoryDataset, with one modification
    # data = data_list[0].__class__()  ->  data = Data()
    # TODO make a PR to PyG?
    @staticmethod
    def collate(data_list: List[Data]) -> Tuple[Data, Dict[str, Tensor]]:
        """
        Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`.
        """
        keys = data_list[0].keys
        # data = data_list[0].__class__()
        data = Data()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if isinstance(item[key], Tensor) and item[key].dim() > 0:
                cat_dim = item.__cat_dim__(key, item[key])
                cat_dim = 0 if cat_dim is None else cat_dim
                s = slices[key][-1] + item[key].size(cat_dim)
            else:
                s = slices[key][-1] + 1
            slices[key].append(s)

        if hasattr(data_list[0], "__num_nodes__"):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            item = data_list[0][key]
            if isinstance(item, Tensor) and len(data_list) > 1:
                if item.dim() > 0:
                    cat_dim = data.__cat_dim__(key, item)
                    cat_dim = 0 if cat_dim is None else cat_dim
                    data[key] = torch.cat(data[key], dim=cat_dim)
                else:
                    data[key] = torch.stack(data[key])
            elif isinstance(item, Tensor):  # Don't duplicate attributes...
                data[key] = data[key][0]
            elif isinstance(item, int) or isinstance(item, float):
                data[key] = torch.tensor(data[key])

            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slice


# this is copied from matmainer.dataset.utils with slight modifications
def _fetch_external_dataset(url, save_dir):
    """
    Downloads file from a given url
    Args:
        url (str): string of where to get external dataset
        save_dir (str): directory where to save the file to be retrieved

    Returns:
        path to the downloaded file
    """

    filename = url.rpartition("/")[2].split("?")[0]
    filepath = to_path(save_dir).joinpath(filename)

    # Fetch data from given url
    msg = "Fetching {} from {} to {}".format(filename, url, filepath)
    print(msg, flush=True)

    r = requests.get(url, stream=True)

    pbar = tqdm.tqdm(
        desc=f"Fetching {url} in MB",
        position=0,
        leave=True,
        ascii=True,
        total=len(r.content),
        unit="MB",
        unit_scale=1e-6,
    )
    chunk_size = 2048
    with open(filepath, "wb") as file_out:
        for chunk in r.iter_content(chunk_size=chunk_size):
            pbar.update(chunk_size)
            file_out.write(chunk)

    r.close()

    return filepath


def _extract_file(filepath, folder):
    """
    Decompress a file.

    Support file types include: .gz .tar.gz, .tgz, .bz2, .zip

    Args:
        filepath: path to the file
        folder: directory to store the extracted file
    """
    filepath = str(to_path(filepath))

    filepath_split = filepath.split(".")
    ext1 = filepath_split[-1].lower()
    ext2 = filepath_split[-2].lower()

    if ext1 == "gz":
        if ext2 == "tar":
            extract_tar(filepath, folder)
        else:
            extract_gz(filepath, folder)
    elif ext1 == "tgz":
        extract_tar(filepath, folder)
    elif ext1 == "bz2":
        extract_bz2(filepath, folder)
    elif ext1 == "zip":
        extract_zip(filepath, folder)
    else:
        # raise ValueError(
        #     f"Cannot extract file {filename}. Unsupported compression format: {ext1}."
        # )

        print(f"No decompression performed for file: {filepath}")

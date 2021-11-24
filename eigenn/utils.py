import inspect
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import torch
import yaml


def to_path(path: Union[str, Path]) -> Path:
    """
    Convert a str to pathlib.Path.
    """
    return Path(path).expanduser().resolve()


def create_directory(path: Union[str, Path], is_directory: bool = False):
    """
    Create the directory for a file.

    Args:
        path: path to the file
        is_directory: whether the file itself is a directory? If yes, will create it;
            if not, will create a directory that is the parent of the file.
    """
    p = to_path(path)

    if is_directory:
        dirname = p
    else:
        dirname = p.parent

    if not dirname.exists():
        os.makedirs(dirname)


def to_list(value: Any) -> Sequence:
    """
    Convert a non-list to a list.
    """
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def yaml_dump(obj, filename: Union[str, Path]):
    """
    Dump an object as yaml.
    """
    create_directory(filename)
    with open(to_path(filename), "w") as f:
        yaml.dump(obj, f, default_flow_style=False)


def yaml_load(filename: Union[str, Path]):
    """
    Load an object from yaml.
    """
    with open(to_path(filename), "r") as f:
        obj = yaml.safe_load(f)

    return obj


def detect_nan_and_inf(x: torch.Tensor, file: Union[str, Path] = None):
    """
    Detect whether a tensor is nan or inf.

    Args:
        x: the tensor
        file: file where this function is called, can be `__file__`.
    """

    def get_line():
        # 0 represents this line
        # 1 represents line at caller
        # 2 represents line at caller of caller
        frame_record = inspect.stack()[2]
        frame = frame_record[0]
        info = inspect.getframeinfo(frame)
        return info.lineno

    if torch.isnan(x):
        raise ValueError(f"Tensor is nan at line {get_line()} of {file}")
    elif torch.isinf(x):
        raise ValueError(f"Tensor is inf at line {get_line()} of {file}")

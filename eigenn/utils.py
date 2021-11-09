import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

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

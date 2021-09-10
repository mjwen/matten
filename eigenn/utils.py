from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union


def to_path(path: Union[str, Path]) -> Path:
    """
    Convert a str to pathlib.Path.
    """
    return Path(path).expanduser().resolve()


def to_list(value: Any) -> Sequence:
    """
    Convert a non-list to a list.
    """
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

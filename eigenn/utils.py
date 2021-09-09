from pathlib import Path
from typing import Union


def to_path(path: Union[str, Path]) -> Path:
    """
    Convert a str to pathlib.Path.
    """
    return Path(path).expanduser().resolve()

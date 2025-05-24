"""Generic utilty functions."""

import pickle
from pathlib import Path
from typing import Any


def load_pickle(file_path: str | Path) -> Any:
    """Load pickle."""
    with open(file_path, "rb") as handle:
        return pickle.load(handle)


def write_pickle(data: Any, file_path: str | Path) -> None:
    """Write data into a pickle file."""
    with open(file_path, "wb") as handle:
        pickle.dump(data, handle)

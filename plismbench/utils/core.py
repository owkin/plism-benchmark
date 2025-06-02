"""Generic utilty functions."""

import os
import pickle
from pathlib import Path
from typing import Any

import requests


def load_pickle(file_path: str | Path) -> Any:
    """Load pickle."""
    with open(file_path, "rb") as handle:
        return pickle.load(handle)


def write_pickle(data: Any, file_path: str | Path) -> None:
    """Write data into a pickle file."""
    with open(file_path, "wb") as handle:
        pickle.dump(data, handle)


def download_state_dict(url: str, name: str) -> str:
    """Download checkpoint from a given URL and store it to disk."""
    output_path = os.path.join(os.environ["HOME"], name)
    if os.path.exists(output_path):
        pass
    else:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error if download fails
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return output_path

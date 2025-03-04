"""Utility functionalities for the extraction pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Do not touch those values as PLISM dataset contains 91 slides x 16278 tiles
NUM_SLIDES: int = 91
NUM_TILES_PER_SLIDE: int = 16_278


def sort_coords(slide_features: np.ndarray) -> np.ndarray:
    """Sort slide features by coordinates."""
    slide_coords = pd.DataFrame(slide_features[:, 1:3], columns=["x", "y"])
    slide_coords.sort_values(["x", "y"], inplace=True)
    new_index = slide_coords.index.values
    return slide_features[new_index]


def save_features(
    slide_features: list[np.ndarray],
    slide_id: str,
    export_path: Path,
) -> None:
    """Save features to disk.

    Parameters
    ----------
    slide_features: list[np.ndarray]
        Current slide features.
    slide_id: str
        Current slide id.
    export_path: Path
        Export path for slide features.
    """
    _output_slide_features = np.concatenate(slide_features, axis=0).astype(np.float32)
    output_slide_features = sort_coords(_output_slide_features)
    slide_num_tiles = output_slide_features.shape[0]
    assert slide_num_tiles == NUM_TILES_PER_SLIDE, (
        f"Output features for slide {slide_id} contains {slide_num_tiles} < {NUM_TILES_PER_SLIDE}."
    )
    np.save(export_path, output_slide_features)


def process_imgs(
    imgs: torch.Tensor, tile_ids: list[str], model: torch.nn.Module
) -> np.ndarray:
    """Perform inference on input (already transformed) images.

    Parameters
    ----------
    imgs: torch.Tensor
        Transformed images (e.g. normalized, cropped, etc.).
    tile_ids: list[str]:
        List of tile ids.
    model: torch.nn.Module
        Feature extractor.
    """
    with torch.inference_mode():
        batch_features = model(imgs).squeeze()  # (N_tiles, d) numpy array
        batch_tiles_coordinates = np.array(
            [tile_id.split("_")[1:] for tile_id in tile_ids]
        ).astype(int)  # (N_tiles, 3) numpy array
    batch_stack = np.concatenate([batch_tiles_coordinates, batch_features], axis=1)
    return batch_stack

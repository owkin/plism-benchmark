"""Stream PLISM tiles dataset and extract features for a given model."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from plismbench.models import FeatureExtractorsEnum


# Do not touch those values as PLISM dataset contains 91 slides x 16278 tiles
NUM_SLIDES: int = 91
NUM_TILES_PER_SLIDE: int = 16_278


class H5Dataset(torch.utils.data.Dataset):
    """Dataset wrapper iterating over a .h5 file content.

    Parameters
    ----------
    file_path: Path
        Path to the .h5 file.
    """

    def __init__(self, file_path: Path):
        super().__init__()
        self.file_path = file_path
        self.data = h5py.File(self.file_path, "r", libver="latest", swmr=True)
        self.keys = list(self.data.keys())

    def __len__(self):
        """Get length of dataset."""
        length = len(self.keys)
        assert length == NUM_TILES_PER_SLIDE, (
            f"H5 file for slide {self.file_path.stem} does not contain {NUM_TILES_PER_SLIDE} tiles!"
        )
        return length

    def __getitem__(self, idx):
        """Get next item (``tile_id``, ``tile_array``)."""
        tile_id = self.keys[idx]
        tile_array = self.data[tile_id][:]
        return tile_id, tile_array


def collate(
    batch: list[tuple[str, torch.Tensor]],
    transform: Callable[[np.ndarray], torch.Tensor],
) -> tuple[list[str], torch.Tensor]:
    """Return tile ids and transformed images.

    Parameters
    ----------
    batch: list[dict[str, Any]]
        List of length ``batch_size`` made of tuples.
        Each tuple represents a tile_id and the corresponding image.
        The image is a torch.float32 tensor (between 0 and 1).
    transform: Callable[[np.ndarray], torch.Tensor]
        Transform function taking ``np.ndarray`` image as inputs.

    Returns
    -------
    output: tuple[list[str], torch.Tensor]
        A tuple made of tiles ids and transformed input images.
    """
    tile_ids: list[str] = [b[0] for b in batch]
    raw_imgs: list[np.ndarray] = [b[1] for b in batch]  # type: ignore
    imgs = torch.stack([transform(img) for img in raw_imgs])
    output = (tile_ids, imgs)
    return output


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


def sort_coords(slide_features: np.ndarray) -> np.ndarray:
    """Sort slide features by coordinates."""
    slide_coords = pd.DataFrame(slide_features[:, 1:3], columns=["x", "y"])
    slide_coords.sort_values(["x", "y"], inplace=True)
    new_index = slide_coords.index.values
    return slide_features[new_index]


def save_features(
    slide_features: list[np.ndarray], slide_id: str, export_path: Path
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


def get_dataloader(
    slide_h5_path: Path,
    transform: Callable[[np.ndarray], torch.Tensor],
    batch_size: int = 32,
    workers: int = 8,
) -> DataLoader:
    """Get PLISM tiles dataset dataloader transformed with ``transform`` function.

    Parameters
    ----------
    slide_h5_path: Path
        Path to the .h5 containing tiles for a given slide.
    transform: Callable[[np.ndarray], torch.Tensor]
        Transform function taking ``np.ndarray`` image as inputs.
    batch_size: int = 32
        Batch size for features extraction.
    workers: int = 8
        Number of workers to load images.

    Returns
    -------
    dataloader: DataLoader
        DataLoader returning (tile_ids, images).
        See ``collate`` function for details.
    """
    dataset = H5Dataset(file_path=slide_h5_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate, transform=transform),
        num_workers=workers,
        pin_memory=True,
        shuffle=False,
    )
    return dataloader


def run_extract(
    feature_extractor_name: str,
    batch_size: int,
    device: int,
    export_dir: Path,
    download_dir: Path,
    overwrite: bool = False,
    workers: int = 8,
) -> None:
    """Run features extraction."""
    if overwrite:
        logger.warning("You are about to overwrite existing features.")
    logger.info(f"Download directory set to {str(download_dir)}.")
    logger.info(f"Export directory set to {str(export_dir)}.")

    # Create export directory if it doesn't exist
    export_dir.mkdir(exist_ok=True, parents=True)

    # Initialize the feature extractor
    feature_extractor = FeatureExtractorsEnum[feature_extractor_name.upper()].init(
        device=device
    )
    image_transform = feature_extractor.transform
    device = feature_extractor.device

    slide_h5_paths = list(download_dir.glob("*_to_GMH_S60.tif.h5"))
    # assert (n_slides := len(slide_h5_paths)) == NUM_SLIDES, (
    #    f"Download uncomplete: found {n_slides}/{NUM_SLIDES}"
    # )

    for slide_h5_path in tqdm(slide_h5_paths):
        # Get slide id
        slide_id = slide_h5_path.stem
        # Get output path for features
        slide_features_export_dir = Path(export_dir / slide_id)
        slide_features_export_path = slide_features_export_dir / "features.npy"
        if slide_features_export_path.exists():
            if overwrite:
                logger.info(
                    f"Features for slide {slide_id} already extracted. Overwriting..."
                )
            else:
                logger.info(
                    f"Features for slide {slide_id} already extracted. Skipping..."
                )
                continue
        slide_features_export_dir.mkdir(exist_ok=True, parents=True)
        # Instanciate the dataloader
        dataloader = get_dataloader(
            slide_h5_path=slide_h5_path,
            transform=image_transform,
            batch_size=batch_size,
            workers=workers,
        )
        # Iterate over the full dataset and store features each time 16,278 input images have been processed
        slide_features: list[np.ndarray] = []
        for tile_ids, tile_images in tqdm(
            dataloader, total=len(dataloader), leave=False
        ):
            batch_stack = process_imgs(tile_images, tile_ids, model=feature_extractor)
            slide_features.append(batch_stack)
        save_features(
            slide_features,
            slide_id=slide_id,
            export_path=slide_features_export_path,
        )
        logger.success(f"Successfully saved features for slide: {slide_id}")

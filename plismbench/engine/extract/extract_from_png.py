"""Stream PLISM tiles dataset and extract features on-the-fly for a given model."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import ceil
from pathlib import Path

import datasets
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from plismbench.engine.extract.utils import (
    NUM_SLIDES,
    NUM_TILES_PER_SLIDE,
    process_imgs,
    save_features,
)
from plismbench.models import FeatureExtractorsEnum


def collate(
    batch: list[dict[str, str | Image.Image]],
    transform: Callable[[np.ndarray], torch.Tensor],
) -> tuple[list[str], list[str], torch.Tensor]:
    """Return slide ids, tile ids and transformed images.

    Parameters
    ----------
    batch: list[dict[str, str | Image.Image]],
        List of length ``batch_size`` made of dictionnaries.
        Each dictionnary is a single input with keys: 'slide_id',
        'tile_id' and 'png'. The image is a ``PIL.Image.Image``
        with type unit8 (0-255)
    transform: Callable[[np.ndarray], torch.Tensor]
        Transform function taking ``np.ndarray`` image as inputs.
        Prior to calling this transform function, conversion from a
        ``PIL.Image.Image`` to an array is performed.

    Returns
    -------
    output: tuple[list[str], list[str], torch.Tensor]
        A tuple made of slides ids, tiles ids and transformed input images.
    """
    slide_ids: list[str] = [b["slide_id"] for b in batch]  #  type: ignore
    tile_ids: list[str] = [b["tile_id"] for b in batch]  #  type: ignore
    imgs = torch.stack([transform(np.array(b["png"])) for b in batch])
    output = (slide_ids, tile_ids, imgs)
    return output


def run_extract_streaming(
    feature_extractor_name: str,
    batch_size: int,
    device: int,
    export_dir: Path,
    overwrite: bool = False,
) -> None:
    """Run features extraction with streaming."""
    if overwrite:
        logger.warning("You are about to overwrite existing features.")
    logger.info(f"Export directory set to {str(export_dir)}.")

    # Create export directory if it doesn't exist
    export_dir.mkdir(exist_ok=True, parents=True)

    # Initialize the feature extractor
    feature_extractor = FeatureExtractorsEnum[feature_extractor_name.upper()].init(
        device=device
    )
    image_transform = feature_extractor.transform
    device = feature_extractor.device

    # Create the dataset and dataloader without actually loading the files to disk (`streaming=True`)
    # The dataset is sorted by slide_id, meaning that the first 16278 indexes belong to the same first slide,
    # then 16278:32556 to the second slide, etc.
    dataset = datasets.load_dataset(
        "owkin/plism-dataset-tiles", split="train", streaming=True
    )
    collate_fn = partial(collate, transform=image_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    # Iterate over the full dataset and store features each time 16278 input images have been processed

    slide_features = []
    current_num_tiles = 0

    existing_slide_already_checked = False
    for slide_ids, tile_ids, imgs in tqdm(
        dataloader,
        total=ceil(NUM_SLIDES * NUM_TILES_PER_SLIDE / batch_size),
        desc="Extracting features",
    ):
        reference_slide_id = slide_ids[0]

        # Get output path for features
        slide_features_export_dir = Path(export_dir / reference_slide_id)
        slide_features_export_path = slide_features_export_dir / "features.npy"

        if slide_features_export_path.exists():
            if not existing_slide_already_checked:
                if overwrite:
                    logger.info(
                        f"Features for slide {reference_slide_id} already extracted. Overwriting..."
                    )
                    existing_slide_already_checked = True
                else:
                    logger.info(
                        f"Features for slide {reference_slide_id} already extracted. Skipping..."
                    )
                    existing_slide_already_checked = True
                    continue
        slide_features_export_dir.mkdir(exist_ok=True, parents=True)

        # If we're on the same slide, we just add the batch features to the running list
        if all(slide_id == reference_slide_id for slide_id in slide_ids):
            batch_stack = process_imgs(imgs, tile_ids, model=feature_extractor)
            slide_features.append(batch_stack)
            # For the very last slide, the last batch may be of size < `batch_size`
            current_num_tiles += batch_stack.shape[0]
            # If the current batch contains exactly the last `batch_size` tile features for the slide,
            # export the slide features and reset `slide_features` and `current_num_tiles`
            if current_num_tiles == NUM_TILES_PER_SLIDE:
                save_features(
                    slide_features,
                    slide_id=reference_slide_id,
                    export_path=slide_features_export_path,
                )
                logger.success(
                    f"Successfully saved features for slide: {reference_slide_id}"
                )
                slide_features = []
                current_num_tiles = 0
                existing_slide_already_checked = False
        # The current batch contains tiles from slide N (`reference_slide_id`) and slide N+1
        else:
            # We retrieve the maximum index at which all tiles in the batch comes from slide N
            mask = np.array(slide_ids) != reference_slide_id
            idx = mask.argmax()
            # And only process the later, then export the slides features
            batch_stack = process_imgs(
                imgs[:idx], tile_ids[:idx], model=feature_extractor
            )
            slide_features.append(batch_stack)
            save_features(
                slide_features,
                slide_id=reference_slide_id,
                export_path=slide_features_export_path,
            )
            logger.success(
                f"Successfully saved features for slide: {reference_slide_id}"
            )
            # We initialize `slide_features` and `current_num_tiles` with respectively
            # the tile features from slide N+1
            slide_features = [
                process_imgs(imgs[idx:], tile_ids[idx:], model=feature_extractor)
            ]
            current_num_tiles = batch_size - idx
            existing_slide_already_checked = False

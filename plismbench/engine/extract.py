
"""Stream PLISM tiles dataset and extract features for a given model."""


from __future__ import annotations
from collections.abc import Callable
import argparse
from pathlib import Path
from PIL import Image
from loguru import logger
from tqdm import tqdm

from functools import partial
from math import ceil
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
from plismbench.models import FeatureExtractorsEnum


def collate(batch: list[dict[str, str | Image.Image]], transform: Callable) -> tuple[list[str], list[str], torch.Tensor]:
    """Return slide ids, tile ids and transformed images.
    
    Parameters
    ----------
    batch: list[dict[str, str | Image.Image]]
        List of length ``batch_size`` made of input dictionnaries.
        Each dictionnary represents an image and contains 3 keys:
        "slide_id", "tile_id", "png". The latter corresponds to the
        input image as ``Image.Image`` type.

    Returns
    -------
    output: tuple[list[str], list[str], torch.Tensor]
        A tuple made of slides ids, tiles ids and transformed input images.
    """
    slide_ids =[b["slide_id"] for b in batch]
    tile_ids = [b["tile_id"] for b in batch]
    imgs = torch.stack([transform(b["png"]) for b in batch], axis=0)
    output = (slide_ids, tile_ids, imgs)
    return output


def process_imgs(imgs: torch.Tensor, tile_ids: list[str], model: torch.nn.Module) -> torch.Tensor:
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
        batch_features = model(imgs.to(device)).squeeze() # (N_tiles, d) numpy array
        batch_tiles_coordinates = np.array([tile_id.split("_")[1:] for tile_id in tile_ids]).astype(int) # (N_tiles, 3) numpy array
    batch_stack = np.concatenate([batch_tiles_coordinates, batch_features], axis=1)
    return batch_stack

def save_features(slide_features: list[np.ndarray], slide_id: str, export_dir: Path) -> None:
    """Save features to disk.
    
    Parameters
    ----------
    slide_features: list[np.ndarray]
        Current slide features.
    slide_id: str
        Current slide id.
    export_dir: Path
        Export root directory.
    """
    slide_features_export_dir = Path(export_dir / slide_id)
    slide_features_export_path = slide_features_export_dir / "features.npy"
    slide_features_export_dir.mkdir(exist_ok=True, parents=True)
    output_slide_features = np.concatenate(slide_features, axis=0).astype(np.float32)
    slide_num_tiles = output_slide_features.shape[0]
    assert slide_num_tiles == NUM_TILES_PER_SLIDES, f"Output features for slide {slide_id} contains {slide_num_tiles} < {NUM_TILES_PER_SLIDES}."
    np.save(slide_features_export_dir, output_slide_features)
    logger.success(f"Successfully saved features for slide: {slide_id}")

def get_dataloader(transform: Callable, batch_size: int = 32) -> DataLoader:
    """Get PLISM tiles dataset dataloader transformed with ``transform`` function.
    
    The dataset and dataloader are not loading the files to disk (`streaming=True`).
    The dataset is sorted by slide_id, meaning that the first 16,278 indexes belong
    to the same first slide, then 16278:32556 to the second slide, etc.

    Parameters
    ----------
    batch_size: int = 32
        Batch size for features extraction.
    
    Returns
    -------
    dataloader: DataLoader
        DataLoader returning (slide_ids, tile_ids, images).
        See ``collate`` function for details.
    """
    dataset = datasets.load_dataset("owkin/plism-dataset-tiles", split="train", streaming=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=partial(collate, transform=transform), num_workers=0, pin_memory=True, shuffle=False
    )
    return dataloader

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extractor", type=str, help="Feature extractor implemented in ``plismbench.models``.", choices=FeatureExtractorsEnum.choices(),
    )
    parser.add_argument(
        "--export-dir", type=Path, help="Export directory.",
    )

    parser.add_argument("--device", type=int, help="Compute device. 0 for cuda:0, -1 for cpu.", default=0)
    parser.add_argument("--batch-size", type=int, help="Batch size for features extraction.", default=64)
    args = parser.parse_args()
    feature_extractor_name = args.extractor
    batch_size = args.batch_size
    export_dir = args.export_dir
    export_dir.mkdir(exist_ok=True, parents=True)

    # You first need to login with your HF token
    from huggingface_hub import login
    login(new_session=False)

    # Initialize the feature extractor
    feature_extractor = FeatureExtractorsEnum[feature_extractor_name.upper()].__init__(device=args.device)
    image_transform = feature_extractor.transform
    device = feature_extractor.device

    # Do not touch those values as PLISM dataset contains 91 slides x 16278 tiles
    NUM_SLIDES = 91
    NUM_TILES_PER_SLIDES = 16_278

    # Instanciate the dataloader
    dataloader = get_dataloader(transform=image_transform, batch_size=batch_size)

    # Iterate over the full dataset and store features each time 16,278 input images have been processed    
    slide_features = []
    current_num_tiles = 0

    for (slide_ids, tile_ids, imgs) in tqdm(
        dataloader,
        total=ceil(NUM_SLIDES * NUM_TILES_PER_SLIDES / batch_size),
        desc="Extracting features"
    ):
        reference_slide_id = slide_ids[0]

        # If we're on the same slide, we just add the batch features to the running list
        if all(slide_id == reference_slide_id for slide_id in slide_ids):
            batch_stack = process_imgs(imgs, tile_ids, model=feature_extractor)
            slide_features.append(batch_stack)
            # For the very last slide, the last batch may be of size < `batch_size`
            current_num_tiles += batch_stack.shape[0]
            # If the current batch contains exactly the last `batch_size` tile features for the slide,
            # export the slide features and reset `slide_features` and `current_num_tiles`
            if current_num_tiles == NUM_TILES_PER_SLIDES:
                save_features(slide_features, slide_id=reference_slide_id, export_dir=export_dir)
                slide_features = []
                current_num_tiles = 0
        # The current batch contains tiles from slide N (`reference_slide_id`) and slide N+1
        else:
            # We retrieve the maximum index at which all tiles in the batch comes from slide N
            mask = (np.array(slide_ids) != reference_slide_id)
            idx = mask.argmax()
            # And only process the later, then export the slides features
            batch_stack = process_imgs(imgs[:idx], tile_ids[:idx], model=feature_extractor)
            slide_features.append(batch_stack)
            save_features(slide_features, slide_id=reference_slide_id, export_dir=export_dir)
            # We initialize `slide_features` and `current_num_tiles` with respectively
            # the tile features from slide N+1
            slide_features = [process_imgs(imgs[idx:], tile_ids[idx:], model=feature_extractor)]
            current_num_tiles = batch_size - idx

"""Utility functions for metrics evaluation."""

import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


NUM_TILES_PER_SLIDE: int = 16_278
NUM_SLIDES: int = 91


def get_tiles_subset_idx(n_tiles: int) -> np.ndarray:
    """Get tiles subset from the original 16_278."""
    if n_tiles == NUM_TILES_PER_SLIDE:
        tiles_subset_idx = np.arange(0, NUM_TILES_PER_SLIDE)
    else:
        tiles_subset_idx = np.load(
            Path(__file__).parents[2] / "assets" / f"tiles_subset_{n_tiles}.npy"
        )
    assert len(set(tiles_subset_idx)) == n_tiles
    return tiles_subset_idx


@lru_cache()
def load_features(fpath: Path) -> np.ndarray:
    """Load features from path using caching and convert to float32."""
    feats = np.load(fpath)
    return feats.astype(np.float32)  # will be converted to float16 later on !


def prepare_features_dataframe(features_dir: Path, extractor: str) -> pd.DataFrame:
    """Prepare unique WSI features dataframe with features paths and metadata."""
    # Get {slide_id: features paths} dictionary
    features_paths = {
        fp: fp.parent.name
        for fp in (features_dir / extractor).glob("*/features.npy")
        if "_to_GMH_S60.tif" in str(fp)
    }
    # Prepare list of slide names, staining, and scanner directly
    slide_data = []
    for features_path, slide_name in features_paths.items():
        staining, scanner = slide_name.split("_")[:2]
        slide_data.append([slide_name, features_path, staining, scanner])

    # Build output dataset
    slide_features = pd.DataFrame(
        slide_data, columns=["slide", "features_path", "staining", "scanner"]
    )
    return slide_features


def prepare_pairs_dataframe(features_dir: Path, extractor: str) -> pd.DataFrame:
    """Prepare all pairs dataframe with features paths and metadata."""
    slide_features = prepare_features_dataframe(
        features_dir=features_dir, extractor=extractor
    )
    assert slide_features.shape == (
        NUM_SLIDES,
        4,
    ), "Slide features dataframe should be of shape (91, 4)."

    pairs = slide_features.merge(slide_features, how="cross", suffixes=("_a", "_b"))
    pairs.set_index(pairs["slide_a"] + "---" + pairs["slide_b"], inplace=True)
    unique_pairs = [
        "---".join([a, b])
        for (a, b) in set(itertools.combinations(slide_features["slide"], 2))
    ]
    pairs = (
        pairs.loc[unique_pairs]  # type: ignore
        .sort_values(["features_path_a", "features_path_b"])
        .reset_index(drop=True)
    )

    assert pairs.shape[0] == int(NUM_SLIDES * (NUM_SLIDES - 1) / 2), (
        "There should be 4,095 unique pairs of slides."
    )
    return pairs

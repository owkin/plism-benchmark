"""Perform features extraction from PLISM dataset."""

from __future__ import annotations

from pathlib import Path

from plismbench.engine.extract.extract_from_h5 import run_extract_h5
from plismbench.engine.extract.extract_from_png import run_extract_streaming


def run_extract(
    feature_extractor_name: str,
    batch_size: int,
    device: int,
    export_dir: Path,
    download_dir: Path | None = None,
    streaming: bool = False,
    overwrite: bool = False,
    workers: int = 8,
) -> None:
    """Run features extraction.

    If ``stream==False``, data will be downloaded and stored to disk from
    https://huggingface.co/datasets/owkin/plism-dataset. This dataset
    contains 91 .h5 files each containing 16,278 images converted
    into numpy arrays. In this scenario, 300Gb storage are necessary.

    If ``stream==True``, data will be downloaded on the fly from
    https://huggingface.co/datasets/owkin/plism-dataset-tiles but not
    stored to disk. This dataset contains 91x16278 images stored as .png
    files. Streaming is enable using the ``datasets`` library and
    `datasets.load_dataset(..., streaming=True)`. Note that this comes
    with the limitation to use ``IterableDataset`` meaning that no easy
    resume can be performed if the features extraction fails.
    """
    if streaming:
        run_extract_streaming(
            feature_extractor_name=feature_extractor_name,
            batch_size=batch_size,
            device=device,
            export_dir=export_dir,
            overwrite=overwrite,
        )
    else:
        assert isinstance(download_dir, str), "Download directory should be specified."
        run_extract_h5(
            feature_extractor_name=feature_extractor_name,
            batch_size=batch_size,
            device=device,
            export_dir=export_dir,
            download_dir=download_dir,
            overwrite=overwrite,
            workers=workers,
        )

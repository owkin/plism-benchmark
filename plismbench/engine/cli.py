"""A module containing CLI commands of the repository."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Union

import typer
from huggingface_hub import login, snapshot_download
from loguru import logger

from plismbench.engine.evaluate import compute_metrics
from plismbench.engine.extract import FeatureExtractorsEnum, run_extract
from plismbench.models.utils import DEFAULT_DEVICE


app = typer.Typer(name="plismbench")


@app.command()
def extract(
    extractor: Annotated[
        str,
        typer.Option(
            "--extractor",
            help="The name of the feature extractor as defined in ``plismbench.models.__init__.py``",
        ),
    ],
    export_dir: Annotated[
        Path,
        typer.Option(
            "--export-dir",
            help=(
                "The root folder where features will be stored."
                " The final export directory is ``export_dir / extractor``"
            ),
        ),
    ],
    download_dir: Annotated[
        Path,
        typer.Option(
            "--download-dir",
            help="Folder containing the .h5 files downloaded from Hugging Face.",
        ),
    ],
    device: Annotated[
        int, typer.Option("--device", help="The CUDA devnumber or -1 for CPU.")
    ] = DEFAULT_DEVICE,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Features extraction batch size.")
    ] = 32,
    workers: Annotated[
        int, typer.Option("--workers", help="Number of workers for async loading.")
    ] = 8,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Whether to overwrite the previous features extraction run.",
        ),
    ] = False,
):
    """Perform features extraction on PLISM histology tiles dataset streamed from Hugging-Face.

    .. code-block:: console

        $ plismbench extract --extractor h0_mini --batch-size 32 --export-dir $HOME/tmp/features/ --download-dir $HOME/tmp/slides/

    """
    if extractor not in FeatureExtractorsEnum.choices():
        raise NotImplementedError(f"Extractor {extractor} not supported.")
    run_extract(
        feature_extractor_name=extractor,
        export_dir=export_dir / extractor,
        download_dir=download_dir,
        device=device,
        batch_size=batch_size,
        workers=workers,
        overwrite=overwrite,
    )


@app.command()
def download(
    download_dir: Annotated[
        Path,
        typer.Option(
            "--download-dir",
            help="Folder containing the .h5 files downloaded from Hugging Face.",
        ),
    ],
    hf_token: Annotated[str, typer.Option("--token", help="Hugging Face token.")],
    workers: Annotated[
        int, typer.Option("--workers", help="Number of workers for parallel download.")
    ] = 8,
):
    """Download PLISM dataset from Hugging Face."""
    login(token=hf_token, new_session=False)
    _ = snapshot_download(
        repo_id="owkin/plism-dataset",
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=["*_to_GMH_S60.tif.h5"],
        ignore_patterns=[".gitattribues"],
        max_workers=workers,
    )


@app.command()
def evaluate(
    extractor: Annotated[
        str,
        typer.Option(
            "--extractor",
            help="The name of the feature extractor as defined in ``plismbench.models.__init__.py``",
        ),
    ],
    features_dir: Annotated[
        Path,
        typer.Option(
            "--features-dir",
            help=(
                "The root folder where features will be stored."
                " The final export directory is ``export_dir / extractor``."
            ),
        ),
    ],
    metrics_dir: Annotated[
        Path,
        typer.Option(
            "--metrics-dir",
            help=(
                "Folder containing the output metrics."
                " The final export directory is ``metrics_dir / extractor``."
            ),
        ),
    ],
    n_tiles: Annotated[
        Union[str, None],
        typer.Option(
            "--n-tiles", help="Number of tiles per slide for metrics computation."
        ),
    ] = None,
    top_k: Annotated[
        Union[str, None],
        typer.Option("--top-k", help="Values of k for top-k accuracy computation."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device", help="'cpu' (parallel computation) or 'gpu' (sequential)."
        ),
    ] = "gpu",
    workers: Annotated[
        int,
        typer.Option(
            "--workers", help="Number of workers for cpu parallel computations."
        ),
    ] = 4,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Whether to overwrite existing metrics.",
        ),
    ] = False,
):
    """Compute robustness metrics for a list of feature extractors."""
    logger.info(f"Computing metrics for extractor {extractor}.")
    _ = compute_metrics(
        features_root_dir=features_dir,
        metrics_save_dir=metrics_dir,
        extractor=extractor,
        top_k=top_k if top_k is None else [int(t) for t in top_k.split(" ")],
        n_tiles=int(n_tiles) if n_tiles is not None else n_tiles,
        device=device,
        overwrite=overwrite,
        workers=workers,
    )


if __name__ == "__main__":
    app()

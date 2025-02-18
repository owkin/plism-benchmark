"""A module containing CLI commands of the repository."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

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
            help="The folder where features will be stored",
        ),
    ],
    device: Annotated[
        int, typer.Option("--device", help="The CUDA devnumber or -1 for CPU.")
    ] = DEFAULT_DEVICE,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Features extraction batch size.")
    ] = 32,
):
    """Tile a slide cohort and store the extracted features."""
    if extractor.upper() not in FeatureExtractorsEnum.choices():
        raise NotImplementedError(f"Extractor {extractor} not supported.")
    run_extract(
        feature_extractor_name=extractor,
        export_dir=export_dir,
        device=device,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    app()

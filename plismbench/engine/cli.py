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
    """Perform features extraction on PLISM histology tiles dataset streamed from Hugging-Face.

    .. code-block:: console

        $ plismbench extract --extractor h0_mini --batch-size 32 --export-dir $HOME/tmp/

    """
    if extractor not in FeatureExtractorsEnum.choices():
        raise NotImplementedError(f"Extractor {extractor} not supported.")
    run_extract(
        feature_extractor_name=extractor,
        export_dir=export_dir,
        device=device,
        batch_size=batch_size,
    )


@app.command()
def login():
    """Login to HuggingFace to download PLISM histology tiles dataset."""
    from huggingface_hub import login

    login(new_session=False)


if __name__ == "__main__":
    app()

"""Models from Slideflow Labs."""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModel

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


class _Mettle15Module(torch.nn.Module):
    """Return METTLE 1.5 refined CLS embeddings as a plain tensor."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute the head-refined CLS embedding (model default output)."""
        return self.model(pixel_values=images, feature_view="cls").pooler_output


class Mettle15(Extractor):
    """METTLE 1.5 model developped by Slideflow Labs available on Hugging-Face (1).

    METTLE 1.5 is a robustness fine-tune of H-optimus-0 (ViT-g/14 with 4 register
    tokens) with a CLS refinement head. Features are the 1536-dimensional
    head-refined CLS embeddings, which is the model's default output.

    .. note::
        (1) https://huggingface.co/slideflow-labs/mettle-1.5

    Parameters
    ----------
    device: int | list[int] | None = DEFAULT_DEVICE,
        Compute resources to use.
        If None, will use all available GPUs.
        If -1, extraction will run on CPU.
    mixed_precision: bool = True
        Whether to use mixed_precision.

    """

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.output_dim = 1536
        self.mixed_precision = mixed_precision

        feature_extractor = _Mettle15Module(
            AutoModel.from_pretrained(
                "slideflow-labs/mettle-1.5", trust_remote_code=True
            )
        )

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )
        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

    @property  # type: ignore
    def transform(self) -> transforms.Compose:
        """Transform method to apply element wise."""
        return transforms.Compose(
            [
                transforms.ToTensor(),  # swap axes and normalize
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Compute and return features.

        Parameters
        ----------
        images: torch.Tensor
            Input of size (n_tiles, n_channels, dim_x, dim_y).

        Returns
        -------
            torch.Tensor: Tensor of size (n_tiles, features_dim).
        """
        features = self.feature_extractor(images.to(self.device))
        return features.cpu().numpy()

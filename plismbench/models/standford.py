"""Models from Stanford University School of Medicine."""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


class PLIP(Extractor):
    """Plip model developped by Stanford University School of Medicine, Stanford, CA (1).

    .. note::
        (1) https://huggingface.co/vinid/plip

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
        self.mixed_precision = mixed_precision

        self.processor = AutoProcessor.from_pretrained("vinid/plip")
        feature_extractor = AutoModelForZeroShotImageClassification.from_pretrained(
            "vinid/plip"
        )

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )
        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

    def process(self, image) -> torch.Tensor:
        """Process input images."""
        plip_input = self.processor(images=image, return_tensors="pt")
        return plip_input["pixel_values"][0]

    @property  # type: ignore
    def transform(self) -> transforms.Lambda:
        """Transform method to apply element wise."""
        return transforms.Lambda(self.process)

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
        if self.mixed_precision:
            features = self.feature_extractor.module.get_image_features(  # type: ignore
                images.to(self.device)
            )
        else:
            features = self.feature_extractor.get_image_features(images.to(self.device))  # type: ignore
        return features.cpu().numpy()

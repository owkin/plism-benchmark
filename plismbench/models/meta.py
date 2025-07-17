"""Models from Meta company."""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


class Dinov2ViTGiant(Extractor):
    """ViT-giant model trained with DINOv2 with 4 registers on ImageNet (1).

    .. note::
        (1) https://github.com/facebookresearch/dinov2?tab=readme-ov-file

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

        feature_extractor = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitg14_reg", verbose=True
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
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
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

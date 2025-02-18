"""Models from Bioptimus company."""

from __future__ import annotations

import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module
from torchvision import transforms



class H0Mini(Extractor):
    """H0-mini model available on Hugging-Face [1]_.

    Parameters
    ----------
    device: int | list[int] | None = DEFAULT_DEVICE,
        Compute resources to use.
        If None, will use all available GPUs.
        If -1, extraction will run on CPU.
    mixed_precision: bool = True
        Whether to use mixed_precision.

    Reference
    ---------
    .. [1] https://huggingface.co/owkin/H0-mini
    """

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super(H0Mini).__init__()
        self.device = device
        self.mixed_precision = mixed_precision

        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
        feature_extractor = timm.create_model(
            "hf-hub:owkin/H0-mini",
            pretrained=True,
            **timm_kwargs
        )
        self.processor = create_transform(
            **resolve_data_config(
                feature_extractor.pretrained_cfg, model=feature_extractor
            )
        )
        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            self.device,
            self.mixed_precision,
        )
        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

    def process(self, image) -> torch.Tensor:
        """Process input images."""
        h0_mini_input = self.processor(image)
        return h0_mini_input

    @property
    def transform(self) -> transforms.Lambda:
        """Transform method to apply element wise."""
        return transforms.Lambda(self.process)


    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Compute and return features.

        Args:
            images (torch.Tensor): Input of size (n_tiles, n_channels, dim_x, dim_y).

        Returns:
            torch.Tensor: Tensor of size (n_tiles, features_dim).
        """
        outputs = self.feature_extractor(images.to(self.device))
        features = outputs.unsqueeze(1).unsqueeze(1)
        return features.cpu().numpy()
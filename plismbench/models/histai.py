"""Models from HistAI company."""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


class HibouBase(Extractor):
    """Hibou-Base model developped by HistAI available on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/histai/hibou-b

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

        self.processor = AutoImageProcessor.from_pretrained(
            "histai/hibou-b", trust_remote_code=True
        )
        feature_extractor = AutoModel.from_pretrained(
            "histai/hibou-b", trust_remote_code=True
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
        hibou_input = self.processor(images=image, return_tensors="pt")
        return hibou_input["pixel_values"][0]

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
        outputs = self.feature_extractor(images.to(self.device))
        features = outputs.last_hidden_state[:, 0, :]
        return features.cpu().numpy()


class HibouLarge(Extractor):
    """Hibou-Large model developped by HistAI available on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/histai/hibou-l

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

        self.processor = AutoImageProcessor.from_pretrained(
            "histai/hibou-L", trust_remote_code=True
        )
        feature_extractor = AutoModel.from_pretrained(
            "histai/hibou-L", trust_remote_code=True
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
        hibou_input = self.processor(images=image, return_tensors="pt")
        return hibou_input["pixel_values"][0]

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
        outputs = self.feature_extractor(images.to(self.device))
        features = outputs.last_hidden_state[:, 0, :]
        return features.cpu().numpy()

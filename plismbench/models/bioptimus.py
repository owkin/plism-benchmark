"""Models from Bioptimus company."""

from __future__ import annotations

from typing import Any

import numpy as np
import timm
import torch
from torchvision import transforms

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


class HOptimus0(Extractor):
    """H-Optimus-0 model developped by Bioptimus available on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/bioptimus/H-optimus-0

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

        timm_kwargs: dict[str, Any] = {
            "init_values": 1e-5,
            "dynamic_img_size": False,
        }
        feature_extractor = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs
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


class H0Mini(Extractor):
    """H0-mini model developped by Owkin & Bioptimus available on Hugging-Face (1).

    You will need to be granted access to be able to use this model.

    .. note::
        (1) https://huggingface.co/bioptimus/H0-mini

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
        self.output_dim = 768
        self.mixed_precision = mixed_precision

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        feature_extractor = timm.create_model(
            "hf-hub:bioptimus/H0-mini", pretrained=True, **timm_kwargs
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
        last_hidden_state = self.feature_extractor(images.to(self.device))
        features = last_hidden_state[:, 0]  # only cls token
        return features.cpu().numpy()


class HOptimus1(Extractor):
    """H-Optimus-1 model developped by Bioptimus available on Hugging-Face (1).

    You will need to be granted access to be able to use this model.

    .. note::
        (1) https://huggingface.co/bioptimus/H-optimus-1

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

        timm_kwargs: dict[str, Any] = {
            "init_values": 1e-5,
            "dynamic_img_size": False,
        }
        feature_extractor = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs
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

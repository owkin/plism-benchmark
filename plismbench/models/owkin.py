"""Models from Owkin, Inc. company."""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


class Phikon(Extractor):
    """Phikon model developped by Owkin available on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/owkin/phikon

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

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        # feature_extractor = ViTModel.from_pretrained(
        #    "owkin/phikon", add_pooling_layer=False
        # )
        feature_extractor = AutoModel.from_pretrained("owkin/phikon")

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )
        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

    def process(self, image) -> torch.Tensor:
        """Process input images."""
        phikon_input = self.processor(images=image, return_tensors="pt")
        return phikon_input["pixel_values"][0]

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
        output = self.feature_extractor(images.to(self.device))
        # If mixed precision is disabled, then the output is a list of
        # 2 items: last_hidden_state and pooler_output.
        # We only extract the hidden state, which is already handled
        # when mixed precision is enabled (see `plismbench.models.utils.MixedPrecisionModule`).
        if len(output) == 2:
            last_hidden_state = output[0]
        else:
            last_hidden_state = output
        features = last_hidden_state[:, 0]
        return features.cpu().numpy()


class PhikonV2(Extractor):
    """Phikon V2 model developped by Owkin available on Hugging-Face (1).

    You will need to be granted access to be able to use this model.

    .. note::
        (1) https://huggingface.co/owkin/phikon-v2

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

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        feature_extractor = AutoModel.from_pretrained("owkin/phikon-v2")

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )
        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

    def process(self, image) -> torch.Tensor:
        """Process input images."""
        phikon_input = self.processor(images=image, return_tensors="pt")
        return phikon_input["pixel_values"][0]

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
        output = self.feature_extractor(images.to(self.device))
        # If mixed precision is disabled, then the output is a list of
        # 2 items: last_hidden_state and pooler_output.
        # We only extract the hidden state, which is already handled
        # when mixed precision is enabled (see `plismbench.models.utils.MixedPrecisionModule`).
        if len(output) == 2:
            last_hidden_state = output[0]
        else:
            last_hidden_state = output
        features = last_hidden_state[:, 0]
        return features.cpu().numpy()

"""Models from Owkin, Inc. company."""

from __future__ import annotations

from typing import Any

import numpy as np
import timm
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


class AquaViT120M105k(Extractor):
    """To be defined."""

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.mixed_precision = mixed_precision

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        feature_extractor = timm.create_model(
            "hf-hub:owkin/aquavit_120M_105k", pretrained=True, **timm_kwargs
        )
        self.output_dim = 768
        self.mixed_precision = mixed_precision

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
        # Concatenate with mean of patch tokens:
        # class_token = outputs[:, 0, :]
        # patch_tokens = output[:, 5:, :]
        # features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        last_hidden_state = self.feature_extractor(images.to(self.device))
        features = last_hidden_state[:, 0]
        return features.cpu().numpy()


class AquaViT120M105kFilteredSubsampled(Extractor):
    """To be defined."""

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.mixed_precision = mixed_precision

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        feature_extractor = timm.create_model(
            "hf-hub:owkin/aquavit_120M_105k_filtered_subsampled",
            pretrained=True,
            **timm_kwargs,
        )
        self.output_dim = 768
        self.mixed_precision = mixed_precision

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
        # Concatenate with mean of patch tokens:
        # class_token = outputs[:, 0, :]
        # patch_tokens = output[:, 5:, :]
        # features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        last_hidden_state = self.feature_extractor(images.to(self.device))
        features = last_hidden_state[:, 0]
        return features.cpu().numpy()


class AquaViT120M315kFilteredSubsampled(Extractor):
    """To be defined."""

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.mixed_precision = mixed_precision

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        feature_extractor = timm.create_model(
            "hf-hub:owkin/aquavit_120M_315k_filtered_subsampled",
            pretrained=True,
            **timm_kwargs,
        )
        self.output_dim = 768
        self.mixed_precision = mixed_precision

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
        # Concatenate with mean of patch tokens:
        # class_token = outputs[:, 0, :]
        # patch_tokens = output[:, 5:, :]
        # features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        last_hidden_state = self.feature_extractor(images.to(self.device))
        features = last_hidden_state[:, 0]
        return features.cpu().numpy()


class AquaViT120M105kFilteredSubsampledHed(Extractor):
    """To be defined."""

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.mixed_precision = mixed_precision

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        feature_extractor = timm.create_model(
            "hf-hub:owkin/aquavit_120M_105k_filtered_subsampled_hed",
            pretrained=True,
            **timm_kwargs,
        )
        self.output_dim = 768
        self.mixed_precision = mixed_precision

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
        # Concatenate with mean of patch tokens:
        # class_token = outputs[:, 0, :]
        # patch_tokens = output[:, 5:, :]
        # features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        last_hidden_state = self.feature_extractor(images.to(self.device))
        features = last_hidden_state[:, 0]
        return features.cpu().numpy()


class MidnightMini120M105kFilteredSubsampled(Extractor):
    """To be defined."""

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.mixed_precision = mixed_precision

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        feature_extractor = timm.create_model(
            "hf-hub:owkin/midnight_mini_120M_105k_filtered_subsampled",
            pretrained=True,
            **timm_kwargs,
        )
        self.output_dim = 768
        self.mixed_precision = mixed_precision

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
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
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
        # Concatenate with mean of patch tokens:
        # class_token = outputs[:, 0, :]
        # patch_tokens = output[:, 5:, :]
        # features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        last_hidden_state = self.feature_extractor(images.to(self.device))
        features = last_hidden_state[:, 0]
        return features.cpu().numpy()


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

        self.output_dim = 768
        self.mixed_precision = mixed_precision

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon")
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
        features = self.feature_extractor(images.to(self.device))
        features = features[:, 0]  # return cls token only
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
        self.output_dim = 1024
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
        last_hidden_state = self.feature_extractor(images.to(self.device))
        features = last_hidden_state[:, 0]
        return features.cpu().numpy()


class DinoV2150k(Extractor):
    """To be defined."""

    def __init__(
        self,
        weights: str,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.mixed_precision = mixed_precision

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        feature_extractor = timm.create_model(
            f"hf-hub:owkin/{weights}",
            pretrained=True,
            **timm_kwargs,
        )
        self.output_dim = 768
        self.mixed_precision = mixed_precision

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
        # Concatenate with mean of patch tokens:
        # class_token = outputs[:, 0, :]
        # patch_tokens = output[:, 5:, :]
        # features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        last_hidden_state = self.feature_extractor(images.to(self.device))
        features = last_hidden_state[:, 0]
        return features.cpu().numpy()

"""Models from Mahmood Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from conch.open_clip_custom import create_model_from_pretrained
from huggingface_hub import snapshot_download
from loguru import logger
from torchvision import transforms
from transformers import AutoModel

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_device, prepare_module


class UNI(Extractor):
    """UNI model developped by Mahmood Lab available on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/MahmoodLab/UNI

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

        timm_kwargs: dict[str, Any] = {
            "init_values": 1e-5,
            "dynamic_img_size": True,
        }
        feature_extractor = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs
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


class UNI2h(Extractor):
    """UNI2-h model developped by Mahmood Lab available on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/MahmoodLab/UNI2-h

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

        timm_kwargs: dict[str, Any] = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        feature_extractor = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
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


class CONCH(Extractor):
    """CONCH model developped by Mahmood Lab available on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/MahmoodLab/CONCH

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

        checkpoint_dir = snapshot_download(repo_id="MahmoodLab/CONCH")
        checkpoint_path = Path(checkpoint_dir) / "pytorch_model.bin"

        feature_extractor, self.processor = create_model_from_pretrained(
            "conch_ViT-B-16",
            force_image_size=224,
            checkpoint_path=str(checkpoint_path),
            device=prepare_device(device),
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
        conch_input = self.processor(image)
        return conch_input

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
        features = self.feature_extractor.encode_image(  # type: ignore
            images.to(self.device), proj_contrast=False, normalize=False
        )
        return features.cpu().numpy()


class CONCHv15(Extractor):
    """Conchv15 model available from TITAN on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/MahmoodLab/conchv1_5
    """

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.mixed_precision = mixed_precision

        titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        feature_extractor, _ = titan.return_conch()

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )
        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

        logger.info("This model is best performing on 448x448 images.")

    @property  # type: ignore
    def transform(self) -> transforms.Lambda:
        """Transform method to apply element wise."""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Compute and return features.

        Args:
            images (torch.Tensor): Input of size (n_tiles, n_channels, dim_x, dim_y).

        Returns
        -------
            torch.Tensor: Tensor of size (n_tiles, features_dim).
        """
        features = self.feature_extractor(images.to(self.device))
        return features.cpu().numpy()

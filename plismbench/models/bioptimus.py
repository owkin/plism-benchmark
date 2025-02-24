"""Models from Bioptimus company."""

from __future__ import annotations

from importlib import resources
from typing import Any

import numpy as np
import timm
import torch
from torchvision import transforms

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, SwiGLUFFNFused, prepare_module


class H0Mini(Extractor):
    """H0-mini model available on Hugging-Face (1).

    .. note::
        (1) https://huggingface.co/owkin/H0-mini

    Parameters
    ----------
    device: int | list[int] | None = DEFAULT_DEVICE,
        Compute resources to use.
        If None, will use all available GPUs.
        If -1, extraction will run on CPU.
    mixed_precision: bool = True
        Whether to use mixed_precision.
    load_from_huggingface: bool = False
        Whether to load from Hugging Face. This parameter is meant
        to ensure anonymity for MICCAI submission. Will be removed
        for the official public release.

    """

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
        load_from_huggingface: bool = False,
    ):
        super().__init__()
        self.mixed_precision = mixed_precision
        self.load_from_huggingface = load_from_huggingface

        if self.load_from_huggingface:
            timm_kwargs: dict[str, Any] = {
                "mlp_layer": timm.layers.SwiGLUPacked,
                "act_layer": torch.nn.SiLU,
            }
            feature_extractor = timm.create_model(
                "hf-hub:bioptimus/H0-mini", pretrained=True, **timm_kwargs
            )
        else:
            # THe default behavior to ensure anonymity.
            timm_kwargs = {
                "model_name": "vit_base_patch14_reg4_dinov2",
                "img_size": 224,
                "patch_size": 14,
                "init_values": 1e-5,
                "num_classes": 0,
                "dynamic_img_size": False,
                "mlp_ratio": 4,
                "mlp_layer": SwiGLUFFNFused,
                "act_layer": torch.nn.SiLU,
            }
            feature_extractor = timm.create_model(**timm_kwargs)
            plism_root_dir = resources.files("plismbench").parent  # type: ignore
            weights = torch.load(
                plism_root_dir / "weights" / "h0_mini.pth",
                weights_only=True,
                map_location="cpu",
            )
            feature_extractor.load_state_dict(weights, strict=False)

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
                transforms.ToTensor(),  # swap axes
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
        if self.load_from_huggingface:
            features = features[:, 0]  # return cls token only
            # Concatenate with mean of patch tokens:
            # class_token = outputs[:, 0, :]
            # patch_tokens = output[:, 5:, :]
            # features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return features.cpu().numpy()

"""Models from Hong Kong University of Science and Technology."""

from __future__ import annotations

import re

import numpy as np
import timm
import torch
from torchvision import transforms

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module
from plismbench.utils.core import download_state_dict


def _convert_state_dict(state_dict: dict) -> dict:
    """Rename state dict keys to match timm's format."""
    state_dict = {
        re.sub(r"blocks\.\d+\.(\d+)", r"blocks.\1", key.replace("backbone.", "")): value
        for key, value in state_dict.items()
    }
    remove_keys = ["mask_token"] + [
        key for key in state_dict.keys() if "dino_head" in key
    ]
    for key in remove_keys:
        state_dict.pop(key)
    return state_dict


class GPFM(Extractor):
    """GPFM model developped by HKUST (1).

    .. note::
        (1)     Ma, J., Guo, Z., Zhou, F., Wang, Y., Xu, Y., et al. (2024).
    Towards a generalizable pathology foundation model via unified knowledge
    distillation (arXiv No. 2407.18449). arXiv. https://arxiv.org/abs/2407.18449

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

        _state_dict_path = download_state_dict(
            url="https://github.com/birkhoffkiki/GPFM/releases/download/ckpt/GPFM.pth",
            name="GPFM.pth",
        )
        _state_dict = torch.load(_state_dict_path, map_location="cpu")
        state_dict = _convert_state_dict(_state_dict["teacher"])

        feature_extractor = timm.create_model(
            model_name="vit_large_patch14_dinov2",
            pretrained=True,
            pretrained_cfg={
                "state_dict": state_dict,
                "num_classes": 0,
            },
            img_size=224,
            patch_size=14,
            init_values=1e-5,
            qkv_bias=True,
            dynamic_img_size=True,
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
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
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

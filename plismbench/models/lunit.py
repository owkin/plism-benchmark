"""Models from Lunit company."""

from __future__ import annotations

import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module
from plismbench.utils.core import download_state_dict


class LunitViTS8(Extractor):
    """ViT-S/8 from Lunit available at (1).

    .. note::
        (1) https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights

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
        self.output_dim = 384
        self.mixed_precision = mixed_precision

        feature_extractor = VisionTransformer(
            img_size=224,
            patch_size=8,
            embed_dim=384,
            num_heads=6,
            num_classes=0,
        )
        state_dict_path = download_state_dict(
            url="https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch8_ep200.torch",
            name="lunit_vit_s8.pth",
        )
        state_dict = torch.load(state_dict_path, map_location="cpu")
        feature_extractor.load_state_dict(state_dict, strict=False)

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
                    mean=(0.70322989, 0.53606487, 0.66096631),
                    std=(0.21716536, 0.26081574, 0.20723464),
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

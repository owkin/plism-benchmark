"""Tests for Meta feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models.extractor import Extractor
from plismbench.models.meta import Dinov2ViTGiant


@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (Dinov2ViTGiant, 1536),
    ],
)
def test_meta_gpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """Meta models test on GPU."""
    meta_model = extractor(device=0)

    x = np.random.rand(224, 224, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")

    transformed_x = meta_model.transform(x)

    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)

    features = meta_model(transformed_x.unsqueeze(0))

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

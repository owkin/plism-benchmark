"""Tests for Owkin, Inc. feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models.owkin import H0Mini


@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (H0Mini, 768),
    ],
)
def test_owkin_cpu(
    extractor: type[H0Mini],
    expected_output_dim: int,
) -> None:
    """Owkin models test on CPU."""
    owkin_model = extractor(device=-1)

    x = np.random.rand(448, 448, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")

    transformed_x = owkin_model.transform(x)

    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)

    features = owkin_model(transformed_x.unsqueeze(0))

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (H0Mini, 768),
    ],
)
def test_owkin_gpu(
    extractor: type[H0Mini],
    expected_output_dim: int,
) -> None:
    """Owkin models test on GPU."""
    x = torch.randn((1, 3, 224, 224))

    owkin_model = extractor(device=0)

    features = owkin_model(x)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

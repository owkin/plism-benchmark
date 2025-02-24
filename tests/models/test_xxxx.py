"""Tests for XXXX feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models.bioptimus import H0Mini


@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (H0Mini, 768),
    ],
)
def test_xxxx_cpu(
    extractor: type[H0Mini],
    expected_output_dim: int,
) -> None:
    """XXXX models test on CPU."""
    xxxx_model = extractor(device=-1)

    x = np.random.rand(224, 224, 3) * 255  # H0Mini works on 224x224 images
    x = Image.fromarray(x.astype("uint8")).convert("RGB")

    transformed_x = xxxx_model.transform(x)

    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)

    features = xxxx_model(transformed_x.unsqueeze(0))

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (H0Mini, 768),
    ],
)
def test_xxxx_gpu(
    extractor: type[H0Mini],
    expected_output_dim: int,
) -> None:
    """XXXX models test on GPU."""
    x = torch.randn((1, 3, 224, 224))  # H0Mini works on 224x224 images

    xxxx_model = extractor(device=0)

    features = xxxx_model(x)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

"""Tests for HistAI feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models.extractor import Extractor
from plismbench.models.histai import HibouBase, HibouLarge


@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (HibouBase, 768),
        (HibouLarge, 1024),
    ],
)
def test_histai_cpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """HistAI models test on CPU."""
    histai_model = extractor(device=-1)

    x = np.random.rand(224, 224, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")

    transformed_x = histai_model.transform(x)

    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)

    features = histai_model(transformed_x.unsqueeze(0))

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (HibouBase, 768),
        (HibouLarge, 1024),
    ],
)
def test_histai_gpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """HistAI models test on GPU."""
    x = torch.randn((1, 3, 224, 224))

    histai_model = extractor(device=0)

    features = histai_model(x)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

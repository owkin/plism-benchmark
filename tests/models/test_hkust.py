"""Tests for Hong Kong University of Science and Technology feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models.extractor import Extractor
from plismbench.models.hkust import GPFM


@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [(GPFM, 1024)],
)
def test_hkust_cpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """HKUST models test on CPU."""
    khust_model = extractor(device=-1)

    x = np.random.rand(224, 224, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")

    transformed_x = khust_model.transform(x)

    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)

    features = khust_model(transformed_x.unsqueeze(0))

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [(GPFM, 1024)],
)
def test_hkust_gpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """HKUST models test on GPU."""
    x = torch.randn((1, 3, 224, 224))

    khust_model = extractor(device=0)

    features = khust_model(x)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

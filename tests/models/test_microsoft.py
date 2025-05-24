"""Tests for Microsoft, Inc. feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models.extractor import Extractor
from plismbench.models.microsoft import ProvGigaPath


@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [(ProvGigaPath, 1536)],
)
def test_microsoft_cpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """Microsoft models test on CPU."""
    microsoft_model = extractor(device=-1)

    x = np.random.rand(224, 224, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")

    transformed_x = microsoft_model.transform(x)

    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)

    features = microsoft_model(transformed_x.unsqueeze(0))

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [(ProvGigaPath, 1536)],
)
def test_microsoft_gpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """Microsoft models test on GPU."""
    x = torch.randn((1, 3, 224, 224))

    microsoft_model = extractor(device=0)

    features = microsoft_model(x)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

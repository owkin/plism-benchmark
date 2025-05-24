"""Tests for Paige AI feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models.extractor import Extractor
from plismbench.models.paige_ai import Virchow, Virchow2


@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (Virchow, 2048),
        (Virchow2, 2560),
    ],
)
def test_paige_cpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """Paige AI models test on CPU."""
    paige_ai_model = extractor(device=-1)

    x = np.random.rand(224, 224, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")

    transformed_x = paige_ai_model.transform(x)

    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)

    features = paige_ai_model(transformed_x.unsqueeze(0))

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (Virchow, 2048),
        (Virchow2, 2560),
    ],
)
def test_paige_gpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """Paige AI models test on GPU."""
    x = torch.randn((1, 3, 224, 224))

    paige_ai_model = extractor(device=0)

    features = paige_ai_model(x)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

"""Tests for Mahmood Lab feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models.extractor import Extractor
from plismbench.models.mahmood_lab import CONCH, UNI, CONCHv15, UNI2h


@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (UNI, 1024),
        (UNI2h, 1536),
        (CONCH, 512),
        (CONCHv15, 768),
    ],
)
def test_mahmood_lab_cpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """Mahmood Lab models test on CPU."""
    mlab_model = extractor(device=-1)

    x = np.random.rand(224, 224, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")

    transformed_x = mlab_model.transform(x)

    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)

    features = mlab_model(transformed_x.unsqueeze(0))

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    [
        (UNI, 1024),
        (UNI2h, 1536),
        (CONCH, 512),
        (CONCHv15, 768),
    ],
)
def test_mahmood_lab_gpu(
    extractor: type[Extractor],
    expected_output_dim: int,
) -> None:
    """Mahmood Lab models test on GPU."""
    x = torch.randn((1, 3, 224, 224))

    mlab_model = extractor(device=0)

    features = mlab_model(x)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

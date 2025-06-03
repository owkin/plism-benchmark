"""Tests feature extractors available in `plismbench`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models import FeatureExtractorsEnum


PARAMETERS = [
    ("h0_mini", 768),
    ("hoptimus0", 1536),
    ("hibou_base", 768),
    ("hibou_large", 1024),
    ("gpfm", 1024),
    ("kaiko_vit_base", 768),
    ("kaiko_vit_large", 1024),
    ("midnight_12k", 3072),
    ("lunit_vit_small_8", 384),
    ("uni", 1024),
    ("uni2h", 1536),
    ("conch", 512),
    ("conchv15", 768),
    ("dinov2_vit_giant", 1536),
    ("provgigapath", 1536),
    ("phikon", 768),
    ("phikonv2", 1024),
    ("virchow", 2560),
    ("virchow2", 2560),
    ("plip", 512),
    # ("hoptimus1", 1536) # access is not granted for now
]


@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    PARAMETERS,
)
def test_extract_cpu(
    extractor: str,
    expected_output_dim: int,
) -> None:
    """Test feature extraction on CPU for all available models."""
    model = FeatureExtractorsEnum[extractor.upper()].init(device=-1)
    # Set a random image and apply transform
    x = np.random.rand(224, 224, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")
    transformed_x = model.transform(x)
    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)
    # Get features
    features = model(transformed_x.unsqueeze(0))
    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    ("extractor", "expected_output_dim"),
    PARAMETERS,
)
def test_extract_gpu(
    extractor: str,
    expected_output_dim: int,
) -> None:
    """Test feature extraction on GPU for all available models."""
    x = torch.randn((1, 3, 224, 224))
    model = FeatureExtractorsEnum[extractor.upper()].init(device=0)
    features = model(x)
    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

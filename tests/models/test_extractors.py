"""Tests feature extractors available in `plismbench`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from plismbench.models import FeatureExtractorsEnum


@pytest.mark.parametrize(
    "extractor",
    ["dinov2_vit_giant_imagenet"],
    # FeatureExtractorsEnum.choices(),
)
def test_extract_cpu(
    extractor: str,
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
    expected_output_dim = model.output_dim
    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    "extractor",
    FeatureExtractorsEnum.choices(),
)
def test_extract_gpu_w_mixed_precision(
    extractor: str,
) -> None:
    """Test feature extraction on GPU for all available models."""
    model = FeatureExtractorsEnum[extractor.upper()].init(device=0)
    # Set a random image and apply transform
    x = np.random.rand(224, 224, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")
    transformed_x = model.transform(x)
    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)
    # Get features
    features = model(transformed_x.unsqueeze(0))
    expected_output_dim = model.output_dim
    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    "extractor",
    FeatureExtractorsEnum.choices(),
)
def test_extract_gpu_wo_mixed_precision(
    extractor: str,
) -> None:
    """Test feature extraction on GPU for all available models."""
    model = FeatureExtractorsEnum[extractor.upper()].init(
        device=0, mixed_precision=False
    )
    # Set a random image and apply transform
    x = np.random.rand(224, 224, 3) * 255
    x = Image.fromarray(x.astype("uint8")).convert("RGB")
    transformed_x = model.transform(x)
    assert isinstance(transformed_x, torch.Tensor)
    assert transformed_x.shape == (3, 224, 224)
    # Get features
    features = model(transformed_x.unsqueeze(0))
    expected_output_dim = model.output_dim
    assert isinstance(features, np.ndarray)
    assert features.shape == (1, expected_output_dim)

"""Test module for cosine similarity metrics."""

import cupy as cp
import numpy as np
import pytest

from plismbench.metrics.cosine_similarity import CosineSimilarity


# Test data
test_data = [
    (
        np.array([[0, 1], [1, 0], [1, 1], [1, 1]]),
        np.array([[0, 2], [2, 0], [-1, -1], [0, 1]]),
        0.426776695,  # Mean cosine similarity should be (1 + 1 - 1 + 1/sqrt(2)) / 4 \approx 0.426776695
    )
]


@pytest.mark.parametrize(("matrix_a", "matrix_b", "expected"), test_data)
def test_cosine_similarity(matrix_a, matrix_b, expected):
    """Test cosine similarity metric."""
    # Test cpu
    metric = CosineSimilarity(device="cpu")
    result = metric.compute_metric(matrix_a, matrix_b)
    assert result == pytest.approx(expected)

    # Check first if a GPU is available
    if cp.cuda.is_available():
        # Test gpu
        metric = CosineSimilarity(device="gpu")
        result = metric.compute_metric(matrix_a, matrix_b)
        assert result == pytest.approx(expected)
    else:
        print("No GPU available. Skipping GPU test.")

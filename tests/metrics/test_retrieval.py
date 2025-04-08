"""Test module for retrieval metrics."""

import numpy as np
import pytest
from loguru import logger

from plismbench.metrics.retrieval import TopkAccuracy


# Test data
test_data = [
    (
        np.array([[0, 1], [1, 0], [0, -1], [-1, 0]]),
        np.array([[0.1, 1.1], [0.2, 1.1], [0.3, 1.1], [0.4, 1.1]]),
        [1, 7],
        np.array([0.125, 1]),
    ),
    (np.random.rand(5, 100), np.random.rand(5, 100), [9], np.array([1])),
]


@pytest.mark.parametrize(("matrix_a", "matrix_b", "k", "expected"), test_data)
def test_topk_accuracy(matrix_a, matrix_b, k, expected):
    """Test top-k accuracy metric."""
    # Test cpu
    metric = TopkAccuracy(device="cpu", k=k)
    result = metric.compute_metric(matrix_a, matrix_b)

    # Check np array equality
    assert result == pytest.approx(expected)


@pytest.mark.local
@pytest.mark.parametrize(("matrix_a", "matrix_b", "k", "expected"), test_data)
def test_topk_accuracy_gpu(matrix_a, matrix_b, k, expected):
    """Test top-k accuracy metric on GPU."""
    import cupy as cp

    # Check first if a GPU is available
    if cp.cuda.is_available():
        # Test gpu
        metric = TopkAccuracy(device="gpu", k=k)
        result = metric.compute_metric(matrix_a, matrix_b)
        assert result == pytest.approx(expected)
    else:
        logger.info("No GPU available. Skipping GPU test.")

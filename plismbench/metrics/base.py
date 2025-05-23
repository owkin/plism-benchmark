"""Module for base metric object."""

from abc import abstractmethod

from loguru import logger


try:
    import cupy as cp
except ImportError as error:
    logger.error(
        f"cupy is not installed. Please run `make install-cupy`.\nError: {error}."
    )
import numpy as np


class BasePlismMetric:
    """Base class for metrics.

    Attributes
    ----------
    device: str: Literal["cpu", "gpu"]
        Device to use for computation.
    """

    def __init__(self, device: str, use_mixed_precision: bool = True):
        self.device = device
        self.ncp = cp if device == "gpu" else np
        self.use_mixed_precision = use_mixed_precision

    @abstractmethod
    def compute_metric(self, matrix_a: np.ndarray, matrix_b: np.ndarray):
        """Compute metric between feature matrices A and B."""
        raise NotImplementedError

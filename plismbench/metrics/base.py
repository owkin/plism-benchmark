"""Module for base metric object."""

from abc import abstractmethod

import cupy as cp
import numpy as np


class BasePlismMetric:
    """Base class for metrics.

    Attributes
    ----------
    device: str: Literal["cpu", "gpu"]
        Device to use for computation.
    """

    def __init__(self, device: str):
        self.device = device
        self.ncp = cp if device == "gpu" else np

    @abstractmethod
    def compute_metric(self, matrix_a: np.ndarray, matrix_b: np.ndarray):
        """Compute metric between feature matrices A and B."""

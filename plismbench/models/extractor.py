"""Core abstract method for feature extractors."""

from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import numpy as np
import torch
from PIL import Image

Transformed = TypeVar("Transformed")


class Extractor(ABC):
    """A base class for :mod:`plismbench` extractors."""

    _feature_extractor: torch.nn.Module

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._transform = lambda x: x

    @property
    def feature_extractor(self) -> torch.nn.Module:
        """
        Feature extractor module

        Returns:
            feature_extractor: torch.nn.Module
        """
        return self._feature_extractor


    @property
    def transform(self) -> Callable[[Image.Image], Transformed]:
        """
        Transform method to apply element wise.
        Default is identity

        Returns
        -------
        transform: Callable[[PIL.Image.Image], Transformed]
        """
        return self._transform


    @abstractmethod
    def __call__(self, images: Transformed) -> np.ndarray:
        """
        Compute and return the MAP features.

        Parameters
        ----------
        images: Transformed
            input of size (N_TILES, N_CHANNELS, DIM_X, DIM_Y). N_TILES=1 for an image

        Returns
        -------
        features : numpy.ndarray
            arrays of size (N_TILES, N_FEATURES) for an image
        """
        raise NotImplementedError

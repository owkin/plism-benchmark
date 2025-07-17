"""Core abstract method for feature extractors."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch


class Extractor(ABC):
    """A base class for :mod:`plismbench` extractors."""

    _feature_extractor: torch.nn.Module
    device: str | torch.device

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._transform = lambda x: x

    @property
    def feature_extractor(self) -> torch.nn.Module:
        """
        Feature extractor module.

        Returns
        -------
            feature_extractor: torch.nn.Module
        """
        return self._feature_extractor

    @feature_extractor.setter
    def feature_extractor(self, feature_extractor_module: torch.nn.Module):
        """Set a new feature extractor module.

        Parameters
        ----------
        feature_extractor_module: feature_extractor_module
        """
        self._feature_extractor = feature_extractor_module

    @property
    def transform(self) -> Callable[[np.ndarray], torch.Tensor]:
        """
        Transform method to apply element wise. Inputs should be np.ndarray.

        This function is applied on ``np.ndarray`` and not ``PIL.Image.Image``
        as HuggingFace data is stored as numpy arrays for pickle checking purposes.
        If your model needs image resizing, then you will need to add a first
        ``transforms.ToPILImage()`` operation, then resizing and finally
        ``transforms.ToTensor()``.
        If your model is best working on images of shape 224x224, then no need
        for rescaling as PLISM tiles have 224x224 shapes.

        Default is identity.

        Returns
        -------
        transform: Callable[[np.ndarray], torch.Tensor]
        """
        return self._transform

    @transform.setter
    def transform(self, transform_function: Callable[[np.ndarray], torch.Tensor]):
        """Set a new transform function to the extractor.

        Parameters
        ----------
        transform_function: Callable[[np.ndarray], Transformed]
            The transform function to be set for the extractor.
        """
        self._transform = transform_function

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """
        Compute and return the MAP features.

        Parameters
        ----------
        images: torch.Tensor
            Input of size (N_TILES, 3, DIM_X, DIM_Y). N_TILES=1 for an image,
            usually DIM_X = DIM_Y = 224.

        Returns
        -------
        features : numpy.ndarray
            arrays of size (N_TILES, N_FEATURES) for an image
        """
        raise NotImplementedError

"""Unit tests for :mod:`plismbench.models`."""

from __future__ import annotations

from enum import Enum

from plismbench.models.bioptimus import H0Mini
from plismbench.models.extractor import Extractor
from plismbench.models.owkin import AquaViT120M105k


class StringEnum(Enum):
    """A base class string enumerator."""

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def choices(cls):
        """Get Enum names."""
        return tuple(i.value for i in cls)


class FeatureExtractorsEnum(StringEnum):
    """A class enumerator for feature extractors."""

    # please follow the format "upper case = lower case"
    # this should map exactly the name in constants
    H0_MINI = "h0_mini"
    AQUAVIT_120M_105K = "aquavit_120M_105k"

    def init(self, device: int | list[int] | None, **kwargs) -> Extractor:
        """Initialize the feature extractor."""
        if self is self.H0_MINI:
            return H0Mini(
                device=device,
                mixed_precision=True,  # don't change this value
                **kwargs,
            )
        elif self is self.AQUAVIT_120M_105K:
            return AquaViT120M105k(
                device=device,
                mixed_precision=True,  # don't change this value
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Extractor {self} is not supported.")

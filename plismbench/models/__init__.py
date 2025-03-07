"""Unit tests for :mod:`plismbench.models`."""

from __future__ import annotations

from enum import Enum

from plismbench.models.bioptimus import H0Mini


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

    def init(self, device: int | list[int] | None, **kwargs):
        """Initialize the feature extractor."""
        if self is self.H0_MINI:
            feature_extractor = H0Mini(
                device=device,
                mixed_precision=True,  # don't change this value
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Extractor {self} is not supported.")
        return feature_extractor

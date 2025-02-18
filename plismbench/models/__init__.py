"""Unit tests for :mod:`plismbench.models`."""

from plismbench.models.owkin import H0Mini
from enum import Enum

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

    def init(self, device: int | list[int] | None):
        """Initialize the feature extractor."""
        if self is self.H0_MINI:
            feature_extractor = H0Mini(
                device=device,
                mixed_precision=True, # don't change this value
            )
        else:
            raise NotImplementedError(f"Extractor {self} is not supported.")
        return feature_extractor

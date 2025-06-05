"""Unit tests for :mod:`plismbench.models`."""

from __future__ import annotations

from enum import Enum

from plismbench.models.bioptimus import H0Mini, HOptimus0, HOptimus1
from plismbench.models.extractor import Extractor
from plismbench.models.histai import HibouBase, HibouLarge
from plismbench.models.hkust import GPFM
from plismbench.models.kaiko_ai import KaikoViTBase, KaikoViTLarge, Midnight12k
from plismbench.models.lunit import LunitViTS8
from plismbench.models.mahmood_lab import CONCH, UNI, CONCHv15, UNI2h
from plismbench.models.meta import Dinov2ViTGiant
from plismbench.models.microsoft import ProvGigaPath
from plismbench.models.owkin import Phikon, PhikonV2
from plismbench.models.paige_ai import Virchow, Virchow2
from plismbench.models.standford import PLIP


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

    # Bioptimus
    H0_MINI = "h0_mini"
    HOPTIMUS0 = "hoptimus0"
    # HOPTIMUS1 = "hoptimus1" # access not granted for now
    # Kaiko AI
    KAIKO_VIT_BASE = "kaiko_vit_base"
    KAIKO_VIT_LARGE = "kaiko_vit_large"
    MIDNIGHT_12K = "midnight_12k"
    # Paige AI
    VIRCHOW = "virchow"
    VIRCHOW2 = "virchow2"
    # Microsoft
    PROVGIGAPATH = "provgigapath"
    # Mahmood Lab
    CONCH = "conch"
    CONCHV15 = "conchv15"
    UNI = "uni"
    UNI2H = "uni2h"
    # HistAI
    HIBOU_BASE = "hibou_base"
    HIBOU_LARGE = "hibou_large"
    # Owkin
    PHIKON = "phikon"
    PHIKONV2 = "phikonv2"
    # HKUST
    GPFM = "gpfm"
    # Standford
    PLIP = "plip"
    # Lunit
    LUNIT_VIT_SMALL_8 = "lunit_vit_small_8"
    # Meta
    DINOV2_VIT_GIANT_IMAGENET = "dinov2_vit_giant_imagenet"

    def init(  # noqa: PLR0911, PLR0912
        self,
        device: int | list[int] | None,
        mixed_precision: bool = True,
        **kwargs,
    ) -> Extractor:
        """Initialize the feature extractor. Mixed precision is set by default."""
        if self is self.H0_MINI:
            return H0Mini(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.HOPTIMUS0:
            return HOptimus0(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        # access not granted for now
        # elif self is self.HOPTIMUS1:
        #     return HOptimus1(
        #         device=device,
        #         mixed_precision=mixed_precision,
        #         **kwargs,
        #     )
        elif self is self.KAIKO_VIT_BASE:
            return KaikoViTBase(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.KAIKO_VIT_LARGE:
            return KaikoViTLarge(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.MIDNIGHT_12K:
            return Midnight12k(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.VIRCHOW:
            return Virchow(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.VIRCHOW2:
            return Virchow2(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.PROVGIGAPATH:
            return ProvGigaPath(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.CONCH:
            return CONCH(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.CONCHV15:
            return CONCHv15(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.UNI:
            return UNI(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.UNI2H:
            return UNI2h(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.HIBOU_BASE:
            return HibouBase(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.HIBOU_LARGE:
            return HibouLarge(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.PHIKON:
            return Phikon(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.PHIKONV2:
            return PhikonV2(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.GPFM:
            return GPFM(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.PLIP:
            return PLIP(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.DINOV2_VIT_GIANT_IMAGENET:
            return Dinov2ViTGiant(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        elif self is self.LUNIT_VIT_SMALL_8:
            return LunitViTS8(
                device=device,
                mixed_precision=mixed_precision,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Extractor {self} is not supported.")

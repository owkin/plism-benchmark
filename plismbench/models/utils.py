"""Utility functions to load and prepare feature extractors."""

from __future__ import annotations

from collections.abc import Callable

import torch
from xformers.ops import SwiGLU


DEFAULT_DEVICE = (
    0 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else -1
)


class MixedPrecisionModule(torch.nn.Module):
    """Mixed Precision Module wrapper.

    Parameters
    ----------
    module: torch.nn.Module
    device_type: str
    """

    def __init__(self, module: torch.nn.Module, device_type: str):
        super(MixedPrecisionModule, self).__init__()
        self.module = module
        self.device_type = device_type

    def forward(self, *args, **kwargs):
        """Forward pass using ``autocast``."""
        # Mixed precision forward
        with torch.amp.autocast(device_type=self.device_type):
            output = self.module(*args, **kwargs)

        if not isinstance(output, torch.Tensor):
            raise ValueError(
                "MixedPrecisionModule currently only supports models returning a single tensor."
            )
        # Back to float32
        return output.to(torch.float32)


def prepare_module(
    module: torch.nn.Module,
    device: int | list[int] | None = None,
    mixed_precision: bool = True,
) -> tuple[torch.nn.Module, str | torch.device]:
    """
    Prepare torch.nn.Module.

    By:
        - setting it to eval mode
        - disabling gradients
        - moving it to the correct device(s)

    Parameters
    ----------
    module: torch.nn.Module
    device: Union[None, int, list[int]] = None
        Compute resources to use.
        If None, will use all available GPUs.
        If -1, extraction will run on CPU.
    mixed_precision: bool = True
        Whether to use mixed_precision (improved throughput on modern GPU cards).

    Returns
    -------
    torch.nn.Module, str | torch.device
    """
    if mixed_precision:
        if not (torch.cuda.is_available() or device == -1):
            raise ValueError("Mixed precision in only available for CUDA GPUs and CPU.")
        module = MixedPrecisionModule(
            module, device_type="cpu" if not torch.cuda.is_available() else "cuda"
        )

    device_: str | torch.device

    if device == -1 or not (
        torch.cuda.is_available() or torch.backends.mps.is_available()
    ):
        device_ = "cpu"
    elif torch.backends.mps.is_available():
        device_ = torch.device("mps")
    elif isinstance(device, int):
        device_ = f"cuda:{device}"
    else:
        # Use DataParallel to distribute the module on all GPUs
        device_ = "cuda:0" if device is None else f"cuda:{device[0]}"
        module = torch.nn.DataParallel(module, device_)  # type: ignore

    module.to(device_)
    module.eval()
    module.requires_grad_(False)

    return module, device_


class SwiGLUFFNFused(SwiGLU):
    """SwiGLUFFNFused layer as implemented in DINO v2 original code base."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., torch.nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )

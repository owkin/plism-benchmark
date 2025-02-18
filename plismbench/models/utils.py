"""Utility functions to load and prepare feature extractors."""

from __future__ import annotations

import torch
from torch.cuda.amp import autocast

DEFAULT_DEVICE = 0 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else -1


class MixedPrecisionModule(torch.nn.Module):
    """Mixed Precision Module wrapper.
    
    Parameters
    ----------
    module: torch.nn.Module
    """
    def __init__(self, module: torch.nn.Module):
        super(MixedPrecisionModule, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """Forward pass using ``autocast``."""
        # Mixed precision forward
        with autocast():
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
) -> tuple[torch.nn.Module, str]:
    """
    Prepare torch.nn.Module by:
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
    torch.nn.Module, str
    """

    if mixed_precision:
        if not (torch.cuda.is_available() or device == -1):
            raise ValueError("Mixed precision in only available for CUDA GPUs and CPU.")
        module = MixedPrecisionModule(module)

    if device == -1 or not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        device = "cpu"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if isinstance(device, int):
            device = f"cuda:{device}"
        else:
            # Use DataParallel to distribute the module on all GPUs
            device = "cuda:0" if device is None else f"cuda:{device[0]}"
            module = torch.nn.DataParallel(module, device)

    module.to(device)
    module.eval()
    module.requires_grad_(False)

    return module, device

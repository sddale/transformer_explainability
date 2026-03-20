"""
Device selection utility — picks the best available accelerator.

Priority: CUDA  →  MPS (Apple Silicon)  →  CPU
"""

import torch


def get_device(force: str | None = None) -> torch.device:
    """
    Return the best available torch device.

    Args:
        force: Override auto-detection.  Accepted values:
               "cuda", "mps", "cpu", or None (auto-detect).

    Returns:
        torch.device
    """
    if force is not None:
        device = torch.device(force)
        print(f"[Device] Forced → {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[Device] CUDA → {name}  ({mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] MPS → Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (no GPU accelerator detected)")

    return device


def is_mps(device: torch.device) -> bool:
    """Check whether *device* is MPS (useful for applying workarounds)."""
    return device.type == "mps"

"""Device abstractions for local execution.

This module provides CPU and GPU device types for local execution,
along with auto-detection utilities.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Literal, cast


class Device(ABC):
    """Base class for compute devices."""

    id: int


@dataclass
class GpuDevice(Device):
    """Single GPU device."""

    id: int

    def __repr__(self):
        return f"GpuDevice({self.id})"


@dataclass
class CpuDevice(Device):
    """CPU device."""

    id: int

    def __repr__(self):
        return f"CpuDevice({self.id})"


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------


def devices(device_type: Literal["cpu", "gpu", "auto"] = "auto") -> list[Device]:
    """Auto-detect available devices on local machine.

    Args:
        device_type: Type of device to detect:
            - "gpu": Return GPU devices only
            - "cpu": Return CPU device only
            - "auto": Return GPUs if available, otherwise CPU (default)

    Returns:
        List of available devices

    Example:
        devs = devices("gpu")   # [GpuDevice(0), GpuDevice(1), ...]
        devs = devices("cpu")   # [CpuDevice(0)]
        devs = devices("auto")  # GPUs if available, else CPU
    """
    if device_type == "cpu":
        return [CpuDevice(id=0)]

    if device_type == "gpu":
        gpus = _detect_gpus()
        if not gpus:
            raise RuntimeError("No GPU devices available")
        return cast(list[Device], gpus)

    # auto mode: prefer GPU, fallback to CPU
    gpus = _detect_gpus()
    if gpus:
        return cast(list[Device], gpus)
    return [CpuDevice(id=0)]


def _detect_gpus() -> list[GpuDevice]:
    """Detect available NVIDIA GPU devices using pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = [GpuDevice(id=i) for i in range(device_count)]
        pynvml.nvmlShutdown()
        return gpus
    except (ImportError, Exception):
        # pynvml not available or no GPUs
        return []

"""Shared validation utilities for distributed components."""


def validate_resources(cpus: int, gpus: int) -> None:
    """Validate resource specifications.

    Args:
        cpus: Number of CPUs
        gpus: Number of GPUs

    Raises:
        ValueError: If resources are invalid
    """
    if cpus < 0 or gpus < 0:
        raise ValueError("Resource counts cannot be negative")
    if cpus == 0 and gpus == 0:
        raise ValueError(
            "At least one resource must be > 0. "
            "Specify cpus=N for CPU-only or gpus=N for GPU-only."
        )

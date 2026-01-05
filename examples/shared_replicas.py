"""Shared replica definitions for distributed examples.

This module should be available on both server and client nodes.
"""

from thinkagain.distributed import replica


@replica(cpus=1)
class DataProcessor:
    """Data processing service."""

    def __init__(self, prefix: str = "default"):
        self.prefix = prefix
        self.call_count = 0

    def process(self, data: str) -> str:
        self.call_count += 1
        return f"{self.prefix}:{data}:count={self.call_count}"

    def get_count(self) -> int:
        return self.call_count


@replica(cpus=1)
class Calculator:
    """Simple calculator service."""

    def __init__(self, multiplier: int = 1):
        self.multiplier = multiplier

    def multiply(self, x: int) -> int:
        return x * self.multiplier

    def add(self, x: int, y: int) -> int:
        return x + y


@replica(cpus=1)
class AsyncService:
    """Async service (e.g., for async model inference)."""

    def __init__(self, delay: float = 0.1):
        self.delay = delay

    async def async_process(self, data: str) -> str:
        import asyncio

        await asyncio.sleep(self.delay)
        return f"processed:{data}"

"""Shared test fixtures."""

from contextlib import contextmanager

import pytest

from thinkagain import node
from thinkagain.distributed import get_default_manager, reset_backend
from thinkagain.distributed.profiling import disable_profiling


# Shared node definitions - now pure functions
@node
async def add_one(x: int) -> int:
    return x + 1


@node
async def double(x: int) -> int:
    return x * 2


@node
async def append_x(logs: list) -> list:
    return logs + ["x"]


@pytest.fixture
def nodes():
    """Return commonly used nodes."""
    return {"add_one": add_one, "double": double, "append_x": append_x}


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure distributed tests start with a clean registry/backend."""
    disable_profiling()
    get_default_manager().clear()
    reset_backend()
    yield
    disable_profiling()
    get_default_manager().clear()
    reset_backend()


@pytest.fixture
def shutdown_on_exit():
    """Context manager to ensure replicas are shutdown."""

    @contextmanager
    def _shutdown(*replicas):
        try:
            yield
        finally:
            for replica_cls in replicas:
                replica_cls.shutdown()

    return _shutdown

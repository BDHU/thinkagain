"""ObjectRef - future for async op results.

An ObjectRef is a reference to a value that may not exist yet. It allows non-blocking
op submission and automatic dependency resolution.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar
from uuid import UUID

__all__ = ["ObjectRef"]

T = TypeVar("T")


@dataclass
class ObjectRef:
    """Reference to a value from an op that may not be computed yet.

    This allows non-blocking op submission where the result can be awaited later.
    ObjectRefs automatically track dependencies and can be passed as arguments to other ops.

    Example:
        ref1 = node_fn.go(x)  # Returns ObjectRef immediately
        ref2 = other_fn.go(ref1)  # Pass ObjectRef as dependency
        result = await ref2  # Block until complete

    Attributes:
        _op_id: Unique identifier for the op producing this value
        _scheduler: Back-reference to the scheduler managing this op
        _result: Cached result once op completes (None while pending)
        _status: Current status of the op
        _completion_event: asyncio.Event signaled when op completes
        _error: Exception if op failed (None if successful)
    """

    _op_id: UUID
    _scheduler: Any  # DAGScheduler (avoid circular import)
    _result: Any | None = None
    _status: Literal["pending", "running", "completed", "failed"] = "pending"
    _completion_event: asyncio.Event = field(default_factory=asyncio.Event)
    _error: Exception | None = None

    async def get(self) -> Any:
        """Wait for op to complete and return the result.

        If the op has already completed, returns the cached result immediately.
        If the op failed, raises the exception from the op.
        Otherwise, blocks until the op completes.

        Returns:
            The result value from the op

        Raises:
            Exception: Whatever exception the op raised, if it failed
        """
        # Fast path: already completed
        if self._status == "completed":
            return self._result

        # Fast path: already failed
        if self._status == "failed":
            assert self._error is not None
            raise self._error

        # Wait for completion
        await self._completion_event.wait()

        # Check status again after waiting
        if self._status == "failed":
            assert self._error is not None
            raise self._error

        return self._result

    def __await__(self):
        """Allow 'await ref' syntax as shorthand for 'await ref.get()'."""
        return self.get().__await__()

    def is_ready(self) -> bool:
        """Check if the result is available without blocking.

        Returns:
            True if the op has completed (successfully or with error)
            False if the op is still pending or running
        """
        return self._status in ("completed", "failed")

    @property
    def status(self) -> Literal["pending", "running", "completed", "failed"]:
        """Get the current status of the op.

        Returns:
            Current op status
        """
        return self._status

    @property
    def op_id(self) -> UUID:
        """Get the unique op ID.

        Returns:
            UUID of the op that produces this value
        """
        return self._op_id

    def __repr__(self) -> str:
        """Human-readable representation."""
        if self._status == "completed":
            return (
                f"ObjectRef({self._op_id}, completed, result={repr(self._result)[:50]})"
            )
        elif self._status == "failed":
            return (
                f"ObjectRef({self._op_id}, failed, error={type(self._error).__name__})"
            )
        else:
            return f"ObjectRef({self._op_id}, {self._status})"

    def __hash__(self) -> int:
        """Make ObjectRef hashable by op_id for use in sets/dicts."""
        return hash(self._op_id)

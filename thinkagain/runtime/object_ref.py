"""ObjectRef - Ray-style future for async task results.

An ObjectRef is a reference to a value that may not exist yet. It allows non-blocking
task submission and automatic dependency resolution.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar
from uuid import UUID

__all__ = ["ObjectRef"]

T = TypeVar("T")


@dataclass
class ObjectRef:
    """Reference to a value from a task that may not be computed yet.

    Similar to Ray's ObjectRef, this allows non-blocking task submission where
    the result can be awaited later. ObjectRefs automatically track dependencies
    and can be passed as arguments to other tasks.

    Example:
        ref1 = node_fn.go(x)  # Returns ObjectRef immediately
        ref2 = other_fn.go(ref1)  # Pass ObjectRef as dependency
        result = await ref2  # Block until complete

    Attributes:
        _task_id: Unique identifier for the task producing this value
        _scheduler: Back-reference to the scheduler managing this task
        _result: Cached result once task completes (None while pending)
        _status: Current status of the task
        _completion_event: asyncio.Event signaled when task completes
        _error: Exception if task failed (None if successful)
    """

    _task_id: UUID
    _scheduler: Any  # DAGScheduler (avoid circular import)
    _result: Any | None = None
    _status: Literal["pending", "running", "completed", "failed"] = "pending"
    _completion_event: asyncio.Event = field(default_factory=asyncio.Event)
    _error: Exception | None = None

    async def get(self) -> Any:
        """Wait for task to complete and return the result.

        If the task has already completed, returns the cached result immediately.
        If the task failed, raises the exception from the task.
        Otherwise, blocks until the task completes.

        Returns:
            The result value from the task

        Raises:
            Exception: Whatever exception the task raised, if it failed
        """
        # Fast path: already completed
        if self._status == "completed":
            return self._result

        # Fast path: already failed
        if self._status == "failed":
            raise self._error  # type: ignore

        # Wait for completion
        await self._completion_event.wait()

        # Check status again after waiting
        if self._status == "failed":
            raise self._error  # type: ignore

        return self._result

    def __await__(self):
        """Allow 'await ref' syntax as shorthand for 'await ref.get()'."""
        return self.get().__await__()

    def is_ready(self) -> bool:
        """Check if the result is available without blocking.

        Returns:
            True if the task has completed (successfully or with error)
            False if the task is still pending or running
        """
        return self._status in ("completed", "failed")

    @property
    def status(self) -> Literal["pending", "running", "completed", "failed"]:
        """Get the current status of the task.

        Returns:
            Current task status
        """
        return self._status

    @property
    def task_id(self) -> UUID:
        """Get the unique task ID.

        Returns:
            UUID of the task that produces this value
        """
        return self._task_id

    def __repr__(self) -> str:
        """Human-readable representation."""
        if self._status == "completed":
            return f"ObjectRef({self._task_id}, completed, result={repr(self._result)[:50]})"
        elif self._status == "failed":
            return f"ObjectRef({self._task_id}, failed, error={type(self._error).__name__})"
        else:
            return f"ObjectRef({self._task_id}, {self._status})"

    def __hash__(self) -> int:
        """Make ObjectRef hashable by task_id for use in sets/dicts."""
        return hash(self._task_id)

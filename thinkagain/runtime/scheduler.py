"""DAGScheduler - runtime scheduler for dynamic op execution."""

import asyncio
import contextvars
from collections.abc import Callable
from typing import Any
from uuid import UUID

from .object_ref import ObjectRef
from .op import ServiceOp, Op
from .utils import resolve_object_refs

__all__ = ["DAGScheduler", "get_current_scheduler", "set_current_scheduler"]

# Context var for thread-safe scheduler access
_scheduler_ctx_var: contextvars.ContextVar["DAGScheduler | None"] = (
    contextvars.ContextVar("scheduler", default=None)
)


def get_current_scheduler() -> "DAGScheduler":
    """Get the scheduler from the current context.

    Returns:
        Current DAGScheduler instance

    Raises:
        RuntimeError: If no scheduler is active (not inside a mesh context)
    """
    scheduler = _scheduler_ctx_var.get()
    if scheduler is None:
        raise RuntimeError(
            "No scheduler active. Ops can only be submitted inside a mesh context: "
            "with mesh: ..."
        )
    return scheduler


def set_current_scheduler(scheduler: "DAGScheduler | None") -> None:
    """Set the scheduler for the current context.

    Args:
        scheduler: Scheduler to set, or None to clear
    """
    _scheduler_ctx_var.set(scheduler)


class DAGScheduler:
    """Runtime scheduler that executes ops as their dependencies resolve.

    The scheduler maintains a dynamic DAG of ops and executes them as soon
    as their dependencies are satisfied. Dependency counts are tracked
    incrementally so scheduling is O(out-degree), avoiding global scans.

    Features:
    - Non-blocking op submission (returns ObjectRef immediately)
    - Automatic dependency resolution
    - Concurrent execution of independent ops
    - Error propagation through ObjectRefs
    - Execution profiling for optimization
    - Ready-queue scheduling (zero CPU usage when idle)

    Example:
        scheduler = DAGScheduler()
        await scheduler.start()

        # Submit ops
        ref1 = scheduler.submit_op(op1)
        ref2 = scheduler.submit_op(op2)  # depends on ref1

        # Wait for results
        result = await ref2

        await scheduler.stop()
    """

    def __init__(self, *, max_concurrency: int | None = None):
        """Initialize the scheduler.

        Args:
            max_concurrency: Maximum number of concurrent ops. If None, ops
                are launched immediately without a worker pool.
        """
        self._ops: dict[UUID, Op] = {}
        self._completed: dict[UUID, Any] = {}
        self._running: set[UUID] = set()
        self._dependents: dict[UUID, set[UUID]] = {}
        self._remaining_deps: dict[UUID, int] = {}
        self._ready_queue: asyncio.Queue[UUID] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._hooks: list[Callable[[Op, tuple, dict], Any]] = []
        self._stopped = False
        self._max_concurrency = max_concurrency

    async def start(self) -> None:
        """Start background workers (if using a worker pool)."""
        if self._workers:
            return

        self._stopped = False
        if self._max_concurrency is None:
            return

        for _ in range(self._max_concurrency):
            self._workers.append(asyncio.create_task(self._worker_loop()))

    async def stop(self) -> None:
        """Stop background workers.

        Cancels workers and waits for them to finish. Any pending ops will not
        be executed. Running ops are not cancelled.
        """
        if not self._workers:
            self._stopped = True
            return

        self._stopped = True
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self._workers = []

    def submit_op(self, op: Op) -> ObjectRef:
        """Submit an op for execution and return an ObjectRef immediately.

        The op will be executed as soon as its dependencies are resolved.

        Args:
            op: Op to execute

        Returns:
            ObjectRef that will contain the result when op completes
        """
        # Create ObjectRef for this op's result
        ref = ObjectRef(_op_id=op.op_id, _scheduler=self)
        op.result_ref = ref

        # Track op and dependencies
        self._ops[op.op_id] = op
        deps = self._unique_dependencies(op)
        remaining = 0
        for dep in deps:
            if dep.is_ready():
                continue
            remaining += 1
            self._dependents.setdefault(dep.op_id, set()).add(op.op_id)
        self._remaining_deps[op.op_id] = remaining

        if remaining == 0:
            self._enqueue_ready(op.op_id)

        return ref

    def register_hook(self, hook: Callable) -> None:
        """Register an execution hook for intercepting op execution.

        Hooks are called before executing an op and can return (True, result)
        to short-circuit execution.

        Args:
            hook: Async callable with signature:
                  async def hook(op, args, kwargs) -> tuple[bool, Any]
        """
        if hook in self._hooks:
            return
        self._hooks.append(hook)

    async def _worker_loop(self) -> None:
        """Worker loop that executes ready ops with bounded concurrency."""
        while not self._stopped:
            op_id = await self._ready_queue.get()
            op = self._ops.get(op_id)
            if op is None or op_id in self._running:
                self._ready_queue.task_done()
                continue

            self._running.add(op_id)
            try:
                await self._execute_op(op)
            finally:
                self._ready_queue.task_done()

    def _enqueue_ready(self, op_id: UUID) -> None:
        """Schedule an op that has no remaining dependencies."""
        if op_id in self._running:
            return
        if self._stopped:
            return
        if self._max_concurrency is None:
            op = self._ops.get(op_id)
            if op is None:
                return
            self._running.add(op_id)
            asyncio.create_task(self._execute_op(op))
            return

        self._ready_queue.put_nowait(op_id)

    def _unique_dependencies(self, op: Op) -> set[ObjectRef]:
        """Return unique ObjectRef dependencies for an op."""
        return set(op.get_dependencies())

    async def _execute_op(self, op: Op) -> None:
        """Execute a single op and store the result in its ObjectRef.

        This handles:
        - Marking op as running
        - Resolving ObjectRef arguments to actual values
        - Executing via hooks or directly
        - Storing result or error in ObjectRef
        - Cleanup

        Args:
            op: Op to execute
        """
        # Mark as running
        if op.result_ref:
            op.result_ref._status = "running"

        try:
            args = await self._resolve_args(op.args)
            kwargs = await self._resolve_kwargs(op.kwargs)
            result = await self._execute_with_hooks(op, args, kwargs)

            self._completed[op.op_id] = result
            if op.result_ref:
                op.result_ref._result = result
                op.result_ref._status = "completed"
                op.result_ref._completion_event.set()

            self._notify_dependents(op.op_id)

        except Exception as e:
            if op.result_ref:
                op.result_ref._error = e
                op.result_ref._status = "failed"
                op.result_ref._completion_event.set()

            self._notify_dependents(op.op_id)

        finally:
            self._running.discard(op.op_id)
            self._ops.pop(op.op_id, None)
            self._remaining_deps.pop(op.op_id, None)

    def _notify_dependents(self, op_id: UUID) -> None:
        """Decrement dependency counts and enqueue newly-ready ops."""
        dependents = self._dependents.pop(op_id, set())
        for dependent_id in dependents:
            if dependent_id not in self._remaining_deps:
                continue
            remaining = self._remaining_deps[dependent_id] - 1
            if remaining <= 0:
                self._remaining_deps[dependent_id] = 0
                self._enqueue_ready(dependent_id)
            else:
                self._remaining_deps[dependent_id] = remaining

    async def _resolve_value(self, value: Any) -> Any:
        """Recursively resolve ObjectRefs to their actual values.

        Handles scalars, ObjectRefs, lists, tuples, and dicts.

        Args:
            value: Any value that may contain ObjectRefs

        Returns:
            Resolved value with all ObjectRefs replaced by their actual values
        """
        return await resolve_object_refs(value)

    async def _resolve_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """Resolve ObjectRefs in arguments to their actual values."""
        resolved = []
        for arg in args:
            resolved.append(await self._resolve_value(arg))
        return tuple(resolved)

    async def _resolve_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve ObjectRefs in keyword arguments to their actual values."""
        resolved = {}
        for k, v in kwargs.items():
            resolved[k] = await self._resolve_value(v)
        return resolved

    async def _execute_with_hooks(self, op: Op, args: tuple, kwargs: dict) -> Any:
        """Execute an op, checking hooks first.

        Hooks can intercept execution (e.g., for distributed execution).
        The first hook that returns (True, result) wins.

        Args:
            op: Op being executed
            args: Resolved arguments
            kwargs: Resolved keyword arguments

        Returns:
            Op result
        """
        # Check hooks in order - pass op object for context
        for hook in self._hooks:
            handled, result = await hook(op, args, kwargs)
            if handled:
                return result

        # Handle ServiceOp specially if no hook handled it
        if isinstance(op, ServiceOp):
            raise RuntimeError(
                f"ServiceOp for {op.service_handle.service_class.__name__}.{op.method_name} "
                f"was not handled by any execution hook. Service execution requires proper "
                f"hook registration. Did you call register_distributed_hooks()?"
            )

        # No hook handled it, execute directly
        if asyncio.iscoroutinefunction(op.fn):
            return await op.fn(*args, **kwargs)
        else:
            return op.fn(*args, **kwargs)

    @property
    def stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with pending, running, and completed counts
        """
        return {
            "pending": max(len(self._tasks) - len(self._running), 0),
            "running": len(self._running),
            "completed": len(self._completed),
        }

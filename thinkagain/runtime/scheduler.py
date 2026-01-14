"""DAGScheduler - runtime scheduler for dynamic task execution."""

import asyncio
import contextvars
from collections.abc import Callable
from typing import Any
from uuid import UUID

from .object_ref import ObjectRef
from .task import ActorTask, Task

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
            "No scheduler active. Tasks can only be submitted inside a mesh context: "
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
    """Runtime scheduler that executes tasks as their dependencies resolve.

    The scheduler maintains a dynamic DAG of tasks and executes them as soon
    as their dependencies are satisfied. Dependency counts are tracked
    incrementally so scheduling is O(out-degree), avoiding global scans.

    Features:
    - Non-blocking task submission (returns ObjectRef immediately)
    - Automatic dependency resolution
    - Concurrent execution of independent tasks
    - Error propagation through ObjectRefs
    - Execution profiling for optimization
    - Ready-queue scheduling (zero CPU usage when idle)

    Example:
        scheduler = DAGScheduler()
        await scheduler.start()

        # Submit tasks
        ref1 = scheduler.submit_task(task1)
        ref2 = scheduler.submit_task(task2)  # depends on ref1

        # Wait for results
        result = await ref2

        await scheduler.stop()
    """

    def __init__(self, *, max_concurrency: int | None = None):
        """Initialize the scheduler.

        Args:
            max_concurrency: Maximum number of concurrent tasks. If None, tasks
                are launched immediately without a worker pool.
        """
        self._tasks: dict[UUID, Task] = {}
        self._completed: dict[UUID, Any] = {}
        self._running: set[UUID] = set()
        self._dependents: dict[UUID, set[UUID]] = {}
        self._remaining_deps: dict[UUID, int] = {}
        self._ready_queue: asyncio.Queue[UUID] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._hooks: list[Callable] = []
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

        Cancels workers and waits for them to finish. Any pending tasks will not
        be executed. Running tasks are not cancelled.
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

    def submit_task(self, task: Task) -> ObjectRef:
        """Submit a task for execution and return an ObjectRef immediately.

        The task will be executed as soon as its dependencies are resolved.

        Args:
            task: Task to execute

        Returns:
            ObjectRef that will contain the result when task completes
        """
        # Create ObjectRef for this task's result
        ref = ObjectRef(_task_id=task.task_id, _scheduler=self)
        task.result_ref = ref

        # Track task and dependencies
        self._tasks[task.task_id] = task
        deps = self._unique_dependencies(task)
        remaining = 0
        for dep in deps:
            if dep.is_ready():
                continue
            remaining += 1
            self._dependents.setdefault(dep.task_id, set()).add(task.task_id)
        self._remaining_deps[task.task_id] = remaining

        if remaining == 0:
            self._enqueue_ready(task.task_id)

        return ref

    def register_hook(self, hook: Callable) -> None:
        """Register an execution hook for intercepting task execution.

        Hooks are called before executing a task and can return (True, result)
        to short-circuit execution.

        Args:
            hook: Async callable with signature:
                  async def hook(fn, args, kwargs) -> tuple[bool, Any]
        """
        if hook not in self._hooks:
            self._hooks.append(hook)

    async def _worker_loop(self) -> None:
        """Worker loop that executes ready tasks with bounded concurrency."""
        while not self._stopped:
            task_id = await self._ready_queue.get()
            task = self._tasks.get(task_id)
            if task is None or task_id in self._running:
                self._ready_queue.task_done()
                continue

            self._running.add(task_id)
            try:
                await self._execute_task(task)
            finally:
                self._ready_queue.task_done()

    def _enqueue_ready(self, task_id: UUID) -> None:
        """Schedule a task that has no remaining dependencies."""
        if task_id in self._running:
            return
        if self._stopped:
            return
        if self._max_concurrency is None:
            task = self._tasks.get(task_id)
            if task is None:
                return
            self._running.add(task_id)
            asyncio.create_task(self._execute_task(task))
            return

        self._ready_queue.put_nowait(task_id)

    def _unique_dependencies(self, task: Task) -> set[ObjectRef]:
        """Return unique ObjectRef dependencies for a task."""
        return set(task.get_dependencies())

    async def _execute_task(self, task: Task) -> None:
        """Execute a single task and store the result in its ObjectRef.

        This handles:
        - Marking task as running
        - Resolving ObjectRef arguments to actual values
        - Executing via hooks or directly
        - Storing result or error in ObjectRef
        - Cleanup

        Args:
            task: Task to execute
        """
        # Mark as running
        if task.result_ref:
            task.result_ref._status = "running"

        try:
            args = await self._resolve_args(task.args)
            kwargs = await self._resolve_kwargs(task.kwargs)
            result = await self._execute_with_hooks(task, args, kwargs)

            self._completed[task.task_id] = result
            if task.result_ref:
                task.result_ref._result = result
                task.result_ref._status = "completed"
                task.result_ref._completion_event.set()

            self._notify_dependents(task.task_id)

        except Exception as e:
            if task.result_ref:
                task.result_ref._error = e
                task.result_ref._status = "failed"
                task.result_ref._completion_event.set()

            self._notify_dependents(task.task_id)

        finally:
            self._running.discard(task.task_id)
            self._tasks.pop(task.task_id, None)
            self._remaining_deps.pop(task.task_id, None)

    def _notify_dependents(self, task_id: UUID) -> None:
        """Decrement dependency counts and enqueue newly-ready tasks."""
        dependents = self._dependents.pop(task_id, set())
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
        if isinstance(value, ObjectRef):
            return await value.get()
        elif isinstance(value, list):
            return [await self._resolve_value(item) for item in value]
        elif isinstance(value, tuple):
            items = [await self._resolve_value(item) for item in value]
            return tuple(items)
        elif isinstance(value, dict):
            result = {}
            for key, val in value.items():
                result[key] = await self._resolve_value(val)
            return result
        else:
            return value

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

    async def _execute_with_hooks(self, task: Task, args: tuple, kwargs: dict) -> Any:
        """Execute a task, checking hooks first.

        Hooks can intercept execution (e.g., for distributed execution).
        The first hook that returns (True, result) wins.

        Args:
            task: Task being executed
            args: Resolved arguments
            kwargs: Resolved keyword arguments

        Returns:
            Task result
        """
        # Check hooks in order - pass task object for context
        for hook in self._hooks:
            # Try to call hook with task parameter (new style)
            import inspect

            sig = inspect.signature(hook)
            if "task" in sig.parameters:
                # New style hook that accepts task parameter
                handled, result = await hook(task, args, kwargs)
            else:
                # Old style hook (for backward compatibility)
                handled, result = await hook(task.fn, args, kwargs)

            if handled:
                return result

        # Handle ActorTask specially if no hook handled it
        if isinstance(task, ActorTask):
            raise RuntimeError(
                f"ActorTask for {task.actor_handle._replica_class.__name__}.{task.method_name} "
                f"was not handled by any execution hook. Actor execution requires proper "
                f"hook registration. Did you call register_distributed_hooks()?"
            )

        # No hook handled it, execute directly
        if asyncio.iscoroutinefunction(task.fn):
            return await task.fn(*args, **kwargs)
        else:
            return task.fn(*args, **kwargs)

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

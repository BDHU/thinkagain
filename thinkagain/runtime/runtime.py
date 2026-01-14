"""Runtime context for dynamic execution."""

from __future__ import annotations

import asyncio
import contextvars

from .scheduler import DAGScheduler
from .task import ActorTask
from .utils import maybe_await


class Runtime:
    """Minimal runtime container for a scheduler and its lifecycle."""

    def __init__(
        self,
        scheduler: DAGScheduler | None = None,
        *,
        enable_local_actor_fallback: bool = False,
    ) -> None:
        self.scheduler = scheduler or DAGScheduler()
        self._started = False
        self._actor_instances: dict[object, object] = {}

        if enable_local_actor_fallback:
            self.scheduler.register_hook(self._local_actor_hook)

    def ensure_started(self) -> None:
        """Start the scheduler loop if an event loop is running."""
        if self._started:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._started = True
        loop.create_task(self.scheduler.start())

    async def _local_actor_hook(
        self, task: object, args: tuple, kwargs: dict
    ) -> tuple[bool, object]:
        """Execute ActorTasks locally when no mesh runtime is active."""
        if not isinstance(task, ActorTask):
            return (False, None)

        actor_handle = task.actor_handle
        instance = self._actor_instances.get(actor_handle)
        if instance is None:
            instance = actor_handle._replica_class(
                *actor_handle._init_args, **actor_handle._init_kwargs
            )
            self._actor_instances[actor_handle] = instance

        method = getattr(instance, task.method_name)
        result = await maybe_await(method, *args, **kwargs)
        return (True, result)


_runtime_ctx_var: contextvars.ContextVar[Runtime | None] = contextvars.ContextVar(
    "runtime", default=None
)
_default_runtime: Runtime | None = None


def _get_default_runtime() -> Runtime:
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = Runtime(enable_local_actor_fallback=True)
    return _default_runtime


def get_current_runtime() -> Runtime:
    """Get the active runtime, falling back to a default local runtime."""
    runtime = _runtime_ctx_var.get()
    if runtime is None:
        return _get_default_runtime()
    return runtime


def set_current_runtime(runtime: Runtime | None) -> contextvars.Token:
    """Set the active runtime for the current context."""
    return _runtime_ctx_var.set(runtime)


def reset_current_runtime(token: contextvars.Token) -> None:
    """Reset the active runtime using a stored token."""
    _runtime_ctx_var.reset(token)

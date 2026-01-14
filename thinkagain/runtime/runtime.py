"""Runtime context for dynamic execution."""

from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING

from .scheduler import DAGScheduler
from .op import ServiceOp
from .utils import maybe_await

if TYPE_CHECKING:
    from ..resources.mesh import Mesh


class Runtime:
    """Minimal runtime container for a scheduler and its lifecycle."""

    def __init__(
        self,
        scheduler: DAGScheduler | None = None,
        *,
        enable_local_service_fallback: bool = False,
    ) -> None:
        self.scheduler = scheduler or DAGScheduler()
        self._started = False
        self._service_instances: dict[object, object] = {}

        if enable_local_service_fallback:
            self.scheduler.register_hook(self._local_service_hook)

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

    async def _local_service_hook(
        self, op: object, args: tuple, kwargs: dict
    ) -> tuple[bool, object]:
        """Execute ServiceOps locally when no mesh runtime is active."""
        if not isinstance(op, ServiceOp):
            return (False, None)

        service_handle = op.service_handle
        instance = self._service_instances.get(service_handle)
        if instance is None:
            instance = service_handle.service_class(
                *service_handle.init_args, **service_handle.init_kwargs
            )
            self._service_instances[service_handle] = instance

        method = getattr(instance, op.method_name)
        result = await maybe_await(method, *args, **kwargs)
        return (True, result)


class RuntimeFactory:
    """Factory for constructing a runtime and scheduler for a mesh."""

    def create_runtime(self, mesh: "Mesh") -> Runtime:
        scheduler = DAGScheduler()
        runtime = Runtime(scheduler)

        from .hooks import dynamic_service_hook

        scheduler.register_hook(dynamic_service_hook)
        return runtime


_runtime_ctx_var: contextvars.ContextVar[Runtime | None] = contextvars.ContextVar(
    "runtime", default=None
)
_default_runtime: Runtime | None = None


def _get_default_runtime() -> Runtime:
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = Runtime(enable_local_service_fallback=True)
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

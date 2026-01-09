"""Replica infrastructure for stateful, replicated actors."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Replica Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicaConfig:
    """Configuration for a replica."""

    gpus: int | None = None
    backend: str = "local"


# ---------------------------------------------------------------------------
# Replica Handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicaHandle:
    """Immutable handle to a replica.

    A handle is a lightweight reference to a replica class and its initialization
    arguments. The actual replica instances are created lazily when the handle is
    deployed on a mesh.
    """

    replica_class_module: str
    replica_class_name: str
    init_args: tuple
    init_kwargs: tuple  # tuple of tuples for hashability
    config: ReplicaConfig
    _uuid: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __hash__(self):
        """Hash by content and UUID for unique identity."""
        return hash(
            (
                self.replica_class_module,
                self.replica_class_name,
                self.init_args,
                self.init_kwargs,
                self._uuid,
            )
        )

    def resolve_class(self) -> type:
        """Resolve class reference at runtime.

        Returns:
            The actual replica class
        """
        import importlib

        module = importlib.import_module(self.replica_class_module)
        return getattr(module, self.replica_class_name)

    def __getattr__(self, method_name: str):
        """Access replica methods.

        Returns a BoundReplicaMethod that can be called during tracing or execution.
        """
        return BoundReplicaMethod(self, method_name)


# ---------------------------------------------------------------------------
# Bound Replica Method
# ---------------------------------------------------------------------------


class BoundReplicaMethod:
    """A bound method on a replica handle.

    This is returned when accessing a method on a ReplicaHandle (e.g., llm.generate).
    When called, it delegates to the registered tracing hook if one exists.
    """

    def __init__(self, handle: ReplicaHandle, method_name: str):
        self.handle = handle
        self.method_name = method_name

    async def __call__(self, *args, **kwargs):
        """Call the replica method.

        Delegates to the registered tracing hook, which may be provided by
        the distributed module or other replica runtimes.
        """
        hook = _get_replica_hook()
        if hook is None:
            raise RuntimeError(
                f"Replica method {self.method_name} called without replica runtime. "
                f"Either:\n"
                f"  1. Call inside @jit function executed within 'with mesh:' block\n"
                f"  2. Register a replica runtime via register_replica_hook()"
            )

        return await hook.record_call(self.method_name, args, kwargs, self.handle)


# ---------------------------------------------------------------------------
# Replica Hook Registry
# ---------------------------------------------------------------------------


_replica_hook: Any = None  # TracingHook protocol


def register_replica_hook(hook: Any) -> None:
    """Register a tracing hook for replica calls.

    This allows the distributed module (or other runtimes) to intercept
    replica method calls and record them appropriately during tracing.

    Args:
        hook: Object implementing the TracingHook protocol
    """
    global _replica_hook
    _replica_hook = hook


def unregister_replica_hook() -> None:
    """Unregister the current replica hook."""
    global _replica_hook
    _replica_hook = None


def _get_replica_hook() -> Any:
    """Get the currently registered replica hook."""
    return _replica_hook


# ---------------------------------------------------------------------------
# Service Decorator
# ---------------------------------------------------------------------------


def replica(gpus: int | None = None, backend: str = "local"):
    """Decorator to mark a class as a stateful replica.

    Replicas are long-lived, stateful components that can be instantiated across
    multiple instances for parallel execution. They are deployed on a mesh and
    called from @jit pipelines.

    Args:
        gpus: Number of GPUs required per replica instance (None for CPU-only)
        backend: Execution backend ("local" or "grpc")

    Example:
        @replica(gpus=1)
        class LLM:
            def __init__(self, model: str):
                self.engine = load_model(model)

            async def generate(self, prompt: str) -> str:
                return await self.engine.generate(prompt)

        # Create handle
        llm = LLM.init("llama-70b")

        # Use in pipeline
        @jit
        async def pipeline(query: str) -> str:
            return await llm.generate(query)

        # Execute with mesh (auto-deploys)
        with mesh:
            result = await pipeline("hello")
    """

    def decorator(cls):
        config = ReplicaConfig(gpus=gpus, backend=backend)

        @classmethod
        def init(cls_, *args, **kwargs) -> ReplicaHandle:
            """Create a replica handle without deploying instances.

            Args:
                *args, **kwargs: Arguments to pass to replica __init__

            Returns:
                ReplicaHandle that can be used in @jit pipelines

            Note:
                Cannot be called inside @jit functions. Create handles
                outside and pass them as pipeline inputs or closures.
            """
            # Check if we're inside tracing (import here to avoid circular dep)
            from .tracing import is_tracing
            from .errors import TracingError

            if is_tracing():
                raise TracingError(
                    f"Cannot call {cls_.__name__}.init() inside @jit. "
                    f"Create replica handles outside @jit functions."
                )

            return ReplicaHandle(
                replica_class_module=cls_.__module__,
                replica_class_name=cls_.__qualname__,
                init_args=args,
                init_kwargs=tuple(sorted(kwargs.items())),
                config=config,
            )

        cls.init = init
        cls._replica_config = config
        return cls

    return decorator

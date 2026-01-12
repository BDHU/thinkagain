"""Replica infrastructure for stateful, replicated actors."""

from __future__ import annotations

import cloudpickle
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from .runtime import maybe_await


# ---------------------------------------------------------------------------
# Replica Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicaConfig:
    """Configuration for a replica."""

    gpus: int | None = None
    backend: str = "local"
    setup: Any | None = None  # Optional setup function for initialization


# ---------------------------------------------------------------------------
# Replica Handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicaHandle:
    """Immutable handle to a replica.

    A handle is a lightweight reference to a replica class and its initialization
    arguments. The actual replica instances are created lazily when the handle is
    deployed on a mesh.

    The class is stored directly (not by name), allowing locally-defined classes
    to work seamlessly. Serialization uses cloudpickle for full class definition
    capture.
    """

    replica_class: type
    init_args: tuple
    init_kwargs: tuple[tuple[str, Any], ...]  # tuple of tuples for hashability
    config: ReplicaConfig
    _uuid: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __hash__(self):
        """Hash by UUID for unique identity.

        Note: We use UUID only since the class itself isn't hashable.
        This means different handles to the same class with same args
        will have different hashes, which is correct for replica identity.
        """
        return hash(self._uuid)

    @property
    def replica_class_name(self) -> str:
        """Get the class name for debugging/error messages."""
        return self.replica_class.__qualname__

    def __reduce__(self):
        """Custom pickle protocol using cloudpickle for the class.

        This allows serialization of locally-defined classes, closures,
        and other Python constructs that standard pickle cannot handle.
        """
        return (
            _reconstruct_handle,
            (
                cloudpickle.dumps(self.replica_class),
                self.init_args,
                self.init_kwargs,
                self.config,
                self._uuid,
            ),
        )

    async def __call__(self, *args, **kwargs):
        """Call the replica directly via its __call__ method.

        This allows handles to be used like: result = await handle(input)
        which delegates to the replica's __call__ method.
        """
        bound_method = BoundReplicaMethod(self)
        return await bound_method(*args, **kwargs)


def _reconstruct_handle(class_bytes, init_args, init_kwargs, config, uuid_str):
    """Reconstruct ReplicaHandle after unpickling.

    This function is called by pickle when deserializing a ReplicaHandle.
    It uses cloudpickle to restore the class definition.

    Args:
        class_bytes: Cloudpickled class definition
        init_args: Initialization arguments
        init_kwargs: Initialization keyword arguments
        config: ReplicaConfig
        uuid_str: UUID string for handle identity

    Returns:
        Reconstructed ReplicaHandle
    """
    replica_class = cloudpickle.loads(class_bytes)
    return ReplicaHandle(
        replica_class=replica_class,
        init_args=init_args,
        init_kwargs=init_kwargs,
        config=config,
        _uuid=uuid_str,
    )


# ---------------------------------------------------------------------------
# Bound Replica Method
# ---------------------------------------------------------------------------


class BoundReplicaMethod:
    """A bound __call__ method on a replica handle.

    This is returned when calling a ReplicaHandle (e.g., await llm(prompt)).
    When called, it delegates to the registered tracing hook if one exists.
    """

    def __init__(self, handle: ReplicaHandle):
        self.handle = handle

    async def __call__(self, *args, **kwargs):
        """Call the replica's __call__ method.

        Replicas must be called from within @node functions, not directly in @jit.
        During execution (when @node body runs), the call is routed through the
        mesh's service provider.
        """
        from ..tracing import is_tracing

        # Replicas should only be called during execution, not during tracing
        # If we're tracing, it means replica was called directly in @jit (Pattern A - wrong)
        # Correct pattern is to call from @node (Pattern B), where body executes later
        if is_tracing():
            from ..errors import TracingError
            from ..graph.graph import TracedValue

            # Get replica name safely (handle might be TracedValue if passed as arg)
            replica_name = (
                self.handle.replica_class_name
                if not isinstance(self.handle, TracedValue)
                else "replica"
            )

            raise TracingError(
                f"Replica '{replica_name}' must be called from within a @node function.\n"
                f"Use this pattern:\n\n"
                f"  @ta.bind_service(svc=handle)\n"
                f"  @ta.node\n"
                f"  async def my_node(x):\n"
                f"      return await svc(x)\n\n"
                f"  @ta.jit\n"
                f"  async def pipeline(x):\n"
                f"      return await my_node(x)"
            )

        # During execution: route through mesh's service provider
        from ...distributed import get_current_mesh

        mesh = get_current_mesh()
        if mesh is None:
            raise RuntimeError(
                f"Replica '{self.handle.replica_class_name}' requires a mesh context. "
                "Use 'with Mesh([...]):' to create a mesh before calling."
            )
        provider = mesh.get_service_provider()
        return await provider.execute_service_call(self.handle, args, kwargs)


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
# Replica State Helpers
# ---------------------------------------------------------------------------


def _iter_slots(cls: type) -> Iterable[str]:
    for base in cls.__mro__:
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot not in ("__dict__", "__weakref__"):
                yield slot


def _copy_replica_state(target: Any, source: Any) -> None:
    if type(target) is not type(source):
        raise TypeError(
            "Replica compose() must return the same class as the target replica."
        )

    if hasattr(target, "__dict__"):
        target.__dict__.clear()
        target.__dict__.update(source.__dict__)

    for slot in _iter_slots(type(target)):
        if hasattr(source, slot):
            setattr(target, slot, getattr(source, slot))


async def apply_replica(replica_obj: Any, fn: Any, *args, **kwargs) -> Any:
    """Apply a @jit-compatible state update to a replica via decompose/compose.

    The replica must implement:
      - decompose(self) -> (children: list[Any] | tuple[Any, ...], aux: Any)
      - compose(cls, aux, children) -> replica instance

    The function must return (new_children, output), where new_children matches
    the structure of decompose()'s children.
    """
    if not hasattr(replica_obj, "decompose") or not hasattr(
        replica_obj.__class__, "compose"
    ):
        raise TypeError(
            "Replica must define decompose() and compose() to use apply_replica()."
        )

    children, aux = replica_obj.decompose()
    if not isinstance(children, (list, tuple)):
        raise TypeError("decompose() must return a list/tuple of children.")

    result = await maybe_await(fn, *children, *args, **kwargs)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise TypeError("Expected (new_children, output) from replica update.")

    new_children, output = result
    if not isinstance(new_children, (list, tuple)):
        new_children = [new_children]
    if len(new_children) != len(children):
        raise ValueError(
            "Updated children count must match decompose() children count."
        )

    updated = type(replica_obj).compose(aux, list(new_children))
    _copy_replica_state(replica_obj, updated)
    return output


# ---------------------------------------------------------------------------
# Service Decorator
# ---------------------------------------------------------------------------


def replica(
    gpus: int | None = None,
    backend: str = "local",
    setup: Any | None = None,
):
    """Decorator to mark a class as a stateful replica.

    Replicas are long-lived, stateful components that can be instantiated across
    multiple instances for parallel execution. They are deployed on a mesh and
    called from @jit pipelines, or served directly with python -m thinkagain.serve.

    The replica class MUST define a __call__ method that serves as the execution
    entry point. This standardizes the interface: handle(input) calls the __call__ method.

    Serialization for Distributed Execution:
        Replica handles use Python's standard pickle protocol (__reduce__) for
        serialization. By default, replica classes are serialized using cloudpickle,
        which handles locally-defined classes and closures.

        For custom serialization behavior (e.g., excluding large models or runtime
        state), implement __reduce__ or __getstate__/__setstate__:

        Example (custom serialization):
            @replica(gpus=1)
            class LLM:
                def __init__(self, model_path: str):
                    self.model_path = model_path
                    self.engine = load_model(model_path)  # Heavy!

                def __reduce__(self):
                    # Only serialize the path, not the loaded engine
                    return (LLM, (self.model_path,))

                async def __call__(self, prompt: str) -> str:
                    return await self.engine.generate(prompt)

        Important: Replica classes should be stateless or serialize their full state.
        State divergence across workers can lead to inconsistent results. If you need
        shared mutable state, use a distributed state store instead.

    Args:
        gpus: Number of GPUs required per replica instance (None for CPU-only)
        backend: Execution backend ("local" or "grpc")
        setup: Optional function to initialize state per instance.
               If provided, the setup function is called once per instance,
               and its return value is passed as the first argument to the
               replicated function.

    Example (stateful class with .init()):
        @replica(gpus=1)
        class LLM:
            def __init__(self, model: str):
                self.engine = load_model(model)

            async def __call__(self, prompt: str) -> str:
                return await self.engine.generate(prompt)

        # Create handle
        llm = LLM.init("llama-70b")

        # Use in pipeline
        @jit
        async def pipeline(query: str) -> str:
            return await llm(query)

        # Execute with mesh (auto-deploys)
        with mesh:
            result = await pipeline("hello")

    Example (direct serving):
        @replica(backend="grpc")
        class TextProcessor:
            def __init__(self):
                self.count = 0

            async def __call__(self, text: str) -> str:
                self.count += 1
                return text.upper()

        # Serve it:
        # python -m thinkagain.serve my_module:TextProcessor --port 8000
    """

    def decorator(cls):
        config = ReplicaConfig(gpus=gpus, backend=backend, setup=setup)

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
            from ..tracing import is_tracing
            from ..errors import TracingError

            if is_tracing():
                raise TracingError(
                    f"Cannot call {cls_.__name__}.init() inside @jit. "
                    f"Create replica handles outside @jit functions."
                )

            return ReplicaHandle(
                replica_class=cls_,
                init_args=args,
                init_kwargs=tuple(sorted(kwargs.items())),
                config=config,
            )

        cls.init = init
        cls._replica_config = config
        return cls

    return decorator

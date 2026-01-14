"""Replica infrastructure for stateful, replicated actors."""

from __future__ import annotations

import cloudpickle
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
    setup: Any | None = None  # Optional setup function for initialization
    name: str | None = None  # Optional human-readable name
    description: str | None = None  # Optional description of the replica


# ---------------------------------------------------------------------------
# Replica Handle
# ---------------------------------------------------------------------------


@dataclass
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
    init_kwargs: dict[str, Any]
    config: ReplicaConfig
    _uuid: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __setattr__(self, name, value):
        """Prevent mutation after initialization."""
        if not hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise TypeError(
                f"Cannot assign to field '{name}' - ReplicaHandle is frozen"
            )

    def __hash__(self):
        """Hash by UUID for unique identity."""
        return hash(self._uuid)

    def __eq__(self, other):
        """Identity-based equality via UUID."""
        if not isinstance(other, ReplicaHandle):
            return False
        return self._uuid == other._uuid

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
    When called, it delegates to the mesh service provider during execution.
    """

    def __init__(self, handle: ReplicaHandle):
        self.handle = handle

    async def __call__(self, *args, **kwargs):
        """Call the replica's __call__ method.

        Routes the call through the mesh's service provider to execute on the
        replica instance.
        """
        # Route through mesh's service provider
        from ..resources import get_current_mesh

        mesh = get_current_mesh()
        if mesh is None:
            raise RuntimeError(
                f"Replica '{self.handle.replica_class_name}' requires a mesh context. "
                "Use 'with Mesh([...]):' to create a mesh before calling."
            )
        provider = mesh.get_service_provider()
        return await provider.execute_service_call(self.handle, args, kwargs)


# ---------------------------------------------------------------------------
# Service Decorator
# ---------------------------------------------------------------------------


def replica(
    gpus: int | None = None,
    backend: str = "local",
    setup: Any | None = None,
    name: str | None = None,
    description: str | None = None,
):
    """Decorator to mark a class as a stateful replica.

    Replicas are long-lived, stateful components that can be instantiated across
    multiple instances for parallel execution. They are deployed on a mesh and
    called from async pipelines, or served directly with python -m thinkagain.serve.

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
        name: Optional human-readable name for the replica (defaults to class name)
        description: Optional description of what the replica does

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
        config = ReplicaConfig(
            gpus=gpus,
            backend=backend,
            setup=setup,
            name=name or cls.__name__,
            description=description,
        )

        @classmethod
        def init(cls_, *args, **kwargs):
            """Create an actor handle with .method.go() support.

            Returns an ActorHandle that supports Ray-style actor API:
                actor = MyClass.init(args)
                ref = actor.method.go(x)
                result = await ref

            Args:
                *args, **kwargs: Arguments to pass to replica __init__

            Returns:
                ActorHandle with .method.go() for each public method

            Example:
                @replica(gpus=1)
                class LLM:
                    def __init__(self, model: str):
                        self.model = model

                    async def generate(self, prompt: str) -> str:
                        return f"Generated: {prompt}"

                llm = LLM.init("llama")

                async def pipeline(query: str):
                    ref = llm.generate.go(query)
                    return await ref
            """
            # Create ActorHandle for dynamic execution
            from .actor import ActorHandle

            return ActorHandle(
                replica_class=cls_,
                init_args=args,
                init_kwargs=kwargs,
                config=config,
            )

        cls.init = init
        cls._replica_config = config
        return cls

    return decorator

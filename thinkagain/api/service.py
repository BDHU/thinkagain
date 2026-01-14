"""Service infrastructure for stateful, distributed services."""

from __future__ import annotations

import cloudpickle
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from ..runtime.object_ref import ObjectRef
from ..runtime import get_current_runtime
from ..runtime.op import ServiceOp

__all__ = [
    "ServiceConfig",
    "ServiceHandle",
    "ServiceClass",
    "RemoteMethod",
    "service",
]


# ---------------------------------------------------------------------------
# Service Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceConfig:
    """Configuration for a service."""

    gpus: int | None = None
    backend: str = "local"
    setup: Any | None = None  # Optional setup function for initialization
    name: str | None = None  # Optional human-readable name
    description: str | None = None  # Optional description of the service


# ---------------------------------------------------------------------------
# Service Handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceHandle:
    """Immutable handle to a service.

    A handle is a lightweight reference to a service class and its initialization
    arguments. The actual service instances are created lazily when the handle is
    deployed on a mesh.

    The class is stored directly (not by name), allowing locally-defined classes
    to work seamlessly. Serialization uses cloudpickle for full class definition
    capture.
    """

    service_class: type = field(compare=False)
    init_args: tuple = field(compare=False)
    init_kwargs: dict[str, Any] = field(compare=False)
    config: ServiceConfig = field(compare=False)
    _uuid: str = field(default_factory=lambda: uuid4().hex, compare=True)

    @property
    def service_class_name(self) -> str:
        """Get the class name for debugging/error messages."""
        return self.service_class.__qualname__

    def __reduce__(self):
        """Custom pickle protocol using cloudpickle for the class.

        This allows serialization of locally-defined classes, closures,
        and other Python constructs that standard pickle cannot handle.
        """
        return (
            _reconstruct_handle,
            (
                cloudpickle.dumps(self.service_class),
                self.init_args,
                self.init_kwargs,
                self.config,
                self._uuid,
            ),
        )


def _reconstruct_handle(class_bytes, init_args, init_kwargs, config, uuid_str):
    """Reconstruct ServiceHandle after unpickling.

    This function is called by pickle when deserializing a ServiceHandle.
    It uses cloudpickle to restore the class definition.

    Args:
        class_bytes: Cloudpickled class definition
        init_args: Initialization arguments
        init_kwargs: Initialization keyword arguments
        config: ServiceConfig
        uuid_str: UUID string for handle identity

    Returns:
        Reconstructed ServiceHandle
    """
    service_class = cloudpickle.loads(class_bytes)
    return ServiceHandle(
        service_class=service_class,
        init_args=init_args,
        init_kwargs=init_kwargs,
        config=config,
        _uuid=uuid_str,
    )


# ---------------------------------------------------------------------------
# Remote Method
# ---------------------------------------------------------------------------


class RemoteMethod:
    """Proxy for service.method.go() calls.

    This enables method calls on services:
        service = MyClass.init(args)
        ref = service.method.go(x)
        result = await ref
    """

    def __init__(self, service_handle: ServiceHandle, method_name: str):
        """Initialize the remote method proxy.

        Args:
            service_handle: The service handle this method belongs to
            method_name: Name of the method to call
        """
        self._service_handle = service_handle
        self._method_name = method_name

    def go(self, *args: Any, **kwargs: Any) -> ObjectRef:
        """Submit method call to scheduler as a service op.

        Returns an ObjectRef immediately. The method will be executed on
        the service instance when its dependencies are ready.

        Args:
            *args: Positional arguments (may contain ObjectRefs)
            **kwargs: Keyword arguments (may contain ObjectRefs)

        Returns:
            ObjectRef that will contain the result when execution completes

        Notes:
            If no runtime context is active, a default local runtime is used.

        Example:
            llm = LLM.init("llama")
            ref = llm.generate.go("hello")
            result = await ref
        """
        runtime = get_current_runtime()
        runtime.ensure_started()

        # Create service op
        op = ServiceOp(
            op_id=uuid4(),
            fn=None,  # Will be resolved from service instance
            args=args,
            kwargs=kwargs,
            service_handle=self._service_handle,
            method_name=self._method_name,
        )

        # Submit and return ObjectRef
        return runtime.scheduler.submit_op(op)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Direct call for backward compatibility.

        This allows calling methods directly without .go():
            result = await service.method(x)

        For dynamic execution, use .go() instead:
            ref = service.method.go(x)
            result = await ref
        """
        # For now, route through .go() for simplicity
        # This could be optimized to bypass scheduler for direct calls
        ref = self.go(*args, **kwargs)
        return await ref

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"RemoteMethod({self._service_handle.service_class.__name__}.{self._method_name})"


# ---------------------------------------------------------------------------
# ServiceClass Handle
# ---------------------------------------------------------------------------


class ServiceClass:
    """Handle to a service with .method.go() support.

    This wraps a ServiceHandle and adds RemoteMethod proxies for each
    public method:

    Example:
        @service(gpus=1)
        class LLM:
            def __init__(self, model: str):
                self.model = model
                self.engine = load_model(model)

            async def generate(self, prompt: str) -> str:
                return await self.engine.generate(prompt)

            async def set_temperature(self, temp: float):
                self.temperature = temp

        # Create service handle
        llm = LLM.init("llama")

        # Call methods with .go()
        async def pipeline(query: str):
            ref1 = llm.generate.go(query)
            result1 = await ref1

            await llm.set_temperature.go(0.9)

            ref2 = llm.generate.go(query)
            result2 = await ref2
            return result2
    """

    def __init__(
        self,
        service_class: type,
        init_args: tuple,
        init_kwargs: dict,
        config: ServiceConfig,
    ):
        """Initialize the service handle.

        Args:
            service_class: The class to instantiate
            init_args: Positional arguments for __init__
            init_kwargs: Keyword arguments for __init__
            config: Service configuration (gpus, backend, etc.)
        """
        self._service_class = service_class
        self._init_args = init_args
        self._init_kwargs = init_kwargs
        self._config = config
        self._method_cache: dict[str, RemoteMethod] = {}

        # Create the underlying ServiceHandle
        self._service_handle = ServiceHandle(
            service_class=service_class,
            init_args=init_args,
            init_kwargs=init_kwargs,
            config=config,
        )

    def _get_remote_method(self, method_name: str) -> RemoteMethod:
        cached = self._method_cache.get(method_name)
        if cached is not None:
            return cached

        if method_name.startswith("_"):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {method_name!r}"
            )

        attr = getattr(self._service_class, method_name, None)
        if attr is None or not callable(attr):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {method_name!r}"
            )

        remote_method = RemoteMethod(self._service_handle, method_name)
        self._method_cache[method_name] = remote_method
        return remote_method

    def __getattr__(self, name: str) -> RemoteMethod:
        """Lazily create RemoteMethod proxies for public methods."""
        return self._get_remote_method(name)

    @property
    def service_handle(self) -> ServiceHandle:
        """Get the underlying ServiceHandle.

        Useful for backward compatibility with code that expects ServiceHandle.
        """
        return self._service_handle

    @property
    def config(self) -> ServiceConfig:
        """Get the service configuration (gpus, backend, etc.)."""
        return self._config

    @property
    def uuid(self) -> UUID:
        """Get the unique ID for this service handle."""
        return UUID(hex=self._service_handle._uuid)

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"ServiceClass({self._service_class.__name__}, uuid={self._service_handle._uuid})"

    def __hash__(self):
        """Hash by UUID for unique identity."""
        return hash(self._service_handle._uuid)


# ---------------------------------------------------------------------------
# Service Decorator
# ---------------------------------------------------------------------------


def service(
    _cls=None,
    *,
    gpus: int | None = None,
    backend: str = "local",
    setup: Any | None = None,
    name: str | None = None,
    description: str | None = None,
):
    """Decorator to mark a class as a stateful service.

    Services are long-lived, stateful components that can be instantiated across
    multiple instances for parallel execution. They are deployed on a mesh and
    called from async pipelines, or served directly with python -m thinkagain.serve.

    Service methods are called using the .go() API: service.method_name.go(args).

    Serialization for Distributed Execution:
        Service handles use Python's standard pickle protocol (__reduce__) for
        serialization. By default, service classes are serialized using cloudpickle,
        which handles locally-defined classes and closures.

        For custom serialization behavior (e.g., excluding large models or runtime
        state), implement __reduce__ or __getstate__/__setstate__:

        Example (custom serialization):
            @service(gpus=1)
            class LLM:
                def __init__(self, model_path: str):
                    self.model_path = model_path
                    self.engine = load_model(model_path)  # Heavy!

                def __reduce__(self):
                    # Only serialize the path, not the loaded engine
                    return (LLM, (self.model_path,))

                async def generate(self, prompt: str) -> str:
                    return await self.engine.generate(prompt)

        Important: Service classes should be stateless or serialize their full state.
        State divergence across workers can lead to inconsistent results. If you need
        shared mutable state, use a distributed state store instead.

    Args:
        gpus: Number of GPUs required per service instance (None for CPU-only)
        backend: Execution backend ("local" or "grpc")
        setup: Optional function to initialize state per instance.
               If provided, the setup function is called once per instance,
               and its return value is passed as the first argument to the
               service function.
        name: Optional human-readable name for the service (defaults to class name)
        description: Optional description of what the service does

    Example (stateful class with .init()):
        @service(gpus=1)
        class LLM:
            def __init__(self, model: str):
                self.engine = load_model(model)

            async def generate(self, prompt: str) -> str:
                return await self.engine.generate(prompt)

        # Create handle
        llm = LLM.init("llama-70b")

        # Use in pipeline
        async def pipeline(query: str) -> str:
            return await llm.generate.go(query)

        # Execute with mesh (auto-deploys)
        with mesh:
            result = await pipeline("hello")

    Example (direct serving):
        @service(backend="grpc")
        class TextProcessor:
            def __init__(self):
                self.count = 0

            async def process(self, text: str) -> str:
                self.count += 1
                return text.upper()

        # Serve it:
        # python -m thinkagain.serve my_module:TextProcessor --port 8000
    """

    def decorator(cls):
        config = ServiceConfig(
            gpus=gpus,
            backend=backend,
            setup=setup,
            name=name or cls.__name__,
            description=description,
        )

        @classmethod
        def init(cls_, *args, **kwargs):
            """Create a service handle with .method.go() support.

            Returns a ServiceClass that supports method calls with .go():
                service = MyClass.init(args)
                ref = service.method.go(x)
                result = await ref

            Args:
                *args, **kwargs: Arguments to pass to service __init__

            Returns:
                ServiceClass with .method.go() for each public method

            Example:
                @service(gpus=1)
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
            # Create ServiceClass for dynamic execution
            return ServiceClass(
                service_class=cls_,
                init_args=args,
                init_kwargs=kwargs,
                config=config,
            )

        cls.init = init
        cls._service_config = config
        return cls

    # Handle @service (no parentheses) - _cls will be the class
    if _cls is not None:
        return decorator(_cls)

    # Handle @service() or @service(gpus=1, etc.) - return decorator function
    return decorator

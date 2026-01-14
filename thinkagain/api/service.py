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
    "ResourceConfig",
    "AutoscalingConfig",
    "ServiceConfig",
    "ServiceHandle",
    "ServiceClass",
    "RemoteMethod",
    "service",
]


# ---------------------------------------------------------------------------
# Service Configuration Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceConfig:
    """Resource requirements per replica.

    Similar to Ray's ray_actor_options, this specifies hardware requirements
    for each service replica.

    Args:
        gpus: Number of GPUs (int) or fraction (float), None for CPU-only
        cpus: Number of CPU cores, None for default

    Example:
        ResourceConfig(gpus=1, cpus=4)
        ResourceConfig(gpus=0.5)  # Fractional GPU
        ResourceConfig()  # CPU-only
    """

    gpus: int | float | None = None
    cpus: int | None = None


@dataclass(frozen=True)
class AutoscalingConfig:
    """Autoscaling bounds and stability controls.

    Similar to Ray Serve's AutoscalingConfig, this configures how the service
    scales in response to load.

    Args:
        min_replicas: Minimum replicas (0 = can scale to zero)
        max_replicas: Maximum replicas (None = unlimited)
        scale_up_delay_s: Seconds to wait before scaling up (default: 10)
        scale_down_delay_s: Seconds to wait before scaling down (default: 300)
        target_concurrent_requests: Target number of concurrent requests per replica.
            - int: User-provided hint for optimal concurrency (e.g., 8 for LLM batching)
            - "auto": Optimizer learns optimal value through profiling (default)

    Example:
        AutoscalingConfig(min_replicas=1, max_replicas=10)
        AutoscalingConfig(min_replicas=0)  # Can scale to zero
        AutoscalingConfig(min_replicas=1, target_concurrent_requests=8)  # Hint
        AutoscalingConfig(target_concurrent_requests="auto")  # Learn optimal
    """

    min_replicas: int = 0
    max_replicas: int | None = None
    scale_up_delay_s: float = 10.0
    scale_down_delay_s: float = 300.0
    target_concurrent_requests: int | str = "auto"


@dataclass(frozen=True)
class ServiceConfig:
    """Service configuration with intelligent auto-scaling.

    The session's optimizer automatically determines:
    - When to scale up/down (based on queue depth, latency, throughput)
    - Which services to prioritize (based on min_replicas and observed behavior)
    - How to allocate scarce resources (based on observed importance)
    """

    resources: ResourceConfig
    autoscaling: AutoscalingConfig
    config: dict[str, Any]
    service_name: str | None = None

    @staticmethod
    def create(
        resources: ResourceConfig | dict[str, Any] | None = None,
        autoscaling: AutoscalingConfig | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "ServiceConfig":
        """Create config with smart defaults.

        Accepts both class instances (ResourceConfig, AutoscalingConfig) and
        dicts for backward compatibility, similar to Ray's API.
        """

        # Convert resources to ResourceConfig if dict
        if isinstance(resources, dict):
            resources_obj = ResourceConfig(**resources)
        elif resources is not None:
            resources_obj = resources
        else:
            resources_obj = ResourceConfig()

        # Convert autoscaling to AutoscalingConfig if dict
        if isinstance(autoscaling, dict):
            autoscaling_obj = AutoscalingConfig(**autoscaling)
        elif autoscaling is not None:
            autoscaling_obj = autoscaling
        else:
            autoscaling_obj = AutoscalingConfig()

        config_dict = config or {}

        return ServiceConfig(
            resources=resources_obj,
            autoscaling=autoscaling_obj,
            config=config_dict,
        )

    @property
    def requires_gpu(self) -> bool:
        """Check if service needs GPU."""
        gpus = self.resources.gpus
        return gpus is not None and gpus > 0

    @property
    def allows_scale_to_zero(self) -> bool:
        """Check if service can scale to zero."""
        return self.autoscaling.min_replicas == 0

    @property
    def requires_guaranteed_capacity(self) -> bool:
        """Check if service needs always-on capacity."""
        return self.autoscaling.min_replicas > 0

    # Legacy properties for backward compatibility with existing code
    @property
    def gpus(self) -> int | float | None:
        """GPU requirement (legacy property)."""
        return self.resources.gpus

    @property
    def backend(self) -> str:
        """Backend type (legacy property, always 'local' now)."""
        return "local"


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
    resources: ResourceConfig | dict[str, Any] | None = None,
    autoscaling: AutoscalingConfig | dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
):
    """Decorator to mark a class as an auto-scaling distributed service.

    The session's optimizer automatically monitors metrics and scales replicas
    to maintain performance while minimizing cost. No manual tuning needed.

    Args:
        resources: Resource requirements per replica (ResourceConfig or dict)
            ResourceConfig(gpus=1, cpus=4) or {"gpus": 1, "cpus": 4}
            - gpus: Number of GPUs (int) or fraction (float), None for CPU-only
            - cpus: Number of CPU cores, None for default

        autoscaling: Scaling bounds and stability controls (AutoscalingConfig or dict)
            AutoscalingConfig(min_replicas=1, max_replicas=10) or
            {"min_replicas": 1, "max_replicas": 10}
            - min_replicas: Minimum replicas (0 = can scale to zero)
            - max_replicas: Maximum replicas (None = unlimited)
            - scale_up_delay_s: Seconds to wait before scaling up (default: 10)
            - scale_down_delay_s: Seconds to wait before scaling down (default: 300)
            - target_concurrent_requests: int or "auto" (default: "auto")
                * int: User hint for optimal concurrency (e.g., 8 for LLM batching)
                * "auto": Optimizer learns through profiling

        config: User-defined runtime config (optional)
            - Passed to __init__(config) if provided
            - Define reconfigure(self, config: dict) for hot reloading

    Examples:
        # Minimal (CPU-only, auto-scaling)
        @service()
        class Cache:
            async def get(self, key: str) -> str:
                return self.data[key]

        # GPU service with class-based config (recommended, Ray-style)
        @service(
            resources=ResourceConfig(gpus=1),
            autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=10),
        )
        class LLM:
            async def generate(self, prompt: str) -> str:
                return await self.engine.generate(prompt)

        # GPU service with dict config (backward compatible)
        @service(
            resources={"gpus": 1},
            autoscaling={"min_replicas": 1, "max_replicas": 10},
        )
        class LLM:
            async def generate(self, prompt: str) -> str:
                return await self.engine.generate(prompt)

        # LLM with batching hint (user knows optimal concurrency is 8)
        @service(
            resources={"gpus": 1},
            autoscaling={
                "min_replicas": 1,
                "max_replicas": 10,
                "target_concurrent_requests": 8,  # Optimal batch size
            },
        )
        class BatchedLLM:
            async def generate(self, prompt: str) -> str:
                return await self.engine.generate(prompt)

        # Let optimizer learn optimal concurrency (default behavior)
        @service(
            resources={"gpus": 1},
            autoscaling={
                "min_replicas": 1,
                "max_replicas": 10,
                "target_concurrent_requests": "auto",  # Explicit auto
            },
        )
        class AutoOptimizedLLM:
            async def generate(self, prompt: str) -> str:
                return await self.engine.generate(prompt)

        # With dynamic config
        @service(
            resources=ResourceConfig(gpus=1),
            config={"temperature": 0.7},
        )
        class ConfigurableLLM:
            def __init__(self, config: dict):
                self.temperature = config["temperature"]

            def reconfigure(self, config: dict):
                self.temperature = config.get("temperature", self.temperature)

            async def generate(self, prompt: str) -> str:
                return await self.engine.generate(
                    prompt,
                    temperature=self.temperature
                )

        # Usage with Session
        mesh = ta.Mesh(devices=[ta.GpuDevice(i) for i in range(8)])
        session = ta.Session(mesh=mesh, optimize="balanced")

        llm = LLM.init()
        with session:
            result = await llm.generate.go("Hello")
    """

    def decorator(cls):
        service_config = ServiceConfig.create(
            resources=resources,
            autoscaling=autoscaling,
            config=config,
        )

        # Set service name
        object.__setattr__(service_config, "service_name", cls.__name__)

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
            """
            return ServiceClass(
                service_class=cls_,
                init_args=args,
                init_kwargs=kwargs,
                config=service_config,
            )

        cls.init = init
        cls._service_config = service_config
        return cls

    # Handle @service (no parentheses) - _cls will be the class
    if _cls is not None:
        return decorator(_cls)

    # Handle @service() or @service(resources={...}, etc.) - return decorator function
    return decorator

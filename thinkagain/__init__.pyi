"""Type stubs for thinkagain - Minimal framework for declarative AI pipelines."""

from typing import Any, Awaitable, Callable, Generic, TypeVar, Union
from dataclasses import dataclass
from uuid import UUID

__version__: str

_T = TypeVar("_T")

@dataclass
class RAGState:
    query: str
    documents: list[str] | None = None
    answer: str = ""
    quality: float = 0.0

class ObjectRef(Generic[_T]):
    """Reference to a value from an op that may not be computed yet."""

    _op_id: UUID
    _result: _T | None
    _status: str
    _error: Exception | None
    async def get(self) -> _T: ...
    def __await__(self): ...
    def is_ready(self) -> bool: ...
    @property
    def done(self) -> bool: ...

class OpType(Generic[_T]):
    """Type returned by @op decorator.

    This type represents a function that can be called directly (await fn(x))
    or submitted to the scheduler (fn.go(x)).
    """
    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[_T]: ...
    def go(self, *args: Any, **kwargs: Any) -> ObjectRef[_T]: ...
    _is_op: bool
    _op_fn: Callable[..., Any]
    _op_name: str
    _op_description: str | None

class ServiceMethod:
    """Protocol for service method proxies with .go()"""
    def go(self, *args: Any, **kwargs: Any) -> ObjectRef[Any]: ...
    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class ServiceClass:
    """Handle to a service with .method.go() support."""

    _service_class: type
    _init_args: tuple
    _init_kwargs: dict
    _config: Any

    def __getattr__(self, name: str) -> ServiceMethod: ...
    @property
    def service_handle(self) -> ServiceHandle: ...
    @property
    def config(self) -> Any: ...
    @property
    def uuid(self) -> UUID: ...

class ServiceConfig:
    gpus: int | None
    backend: str
    setup: Any | None
    name: str | None
    description: str | None

class ServiceHandle:
    """Immutable handle to a service."""

    service_class: type
    init_args: tuple
    init_kwargs: dict[str, Any]
    config: ServiceConfig
    _uuid: str

    @property
    def service_class_name(self) -> str: ...

class Device:
    id: int

@dataclass
class CpuDevice(Device):
    id: int
    def __repr__(self) -> str: ...

@dataclass
class GpuDevice(Device):
    id: int
    def __repr__(self) -> str: ...

@dataclass
class MeshNode:
    name: str
    gpus: int = 0
    cpus: int = 0
    endpoint: str | None = None

    def expand(self) -> list[Device]: ...

class Mesh:
    """Resource mesh defining available compute for distributed execution."""

    backend: str
    devices: list[Device]
    total_gpus: int
    total_cpus: int

    def __init__(self, devices: list[Device | MeshNode], backend: str = "local"): ...
    def __enter__(self) -> Mesh: ...
    def __exit__(self, *args: Any) -> None: ...
    def max_instances(self, gpus: int | None) -> int | float: ...
    def get_endpoints(self) -> list[str]: ...
    def get_service_provider(self) -> Any: ...

def devices(device_type: str = "auto") -> list[Device]: ...
def op(
    fn: Callable[..., _T] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Union[
    Callable[[Callable[..., _T]], OpType[_T]],
    OpType[_T],
]: ...
def service(
    gpus: int | None = None,
    backend: str = "local",
    setup: Any | None = None,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[type], type]: ...
def get_current_mesh() -> Mesh | None: ...
def require_mesh(context: str = "This operation") -> Mesh: ...
def register_distributed_hooks() -> None: ...
def unregister_distributed_hooks() -> None: ...
def enable_profiling() -> Any: ...
def disable_profiling() -> None: ...
def is_profiling_enabled() -> bool: ...
def get_profiler() -> Any | None: ...
def profile(max_samples: int = 10_000) -> Any: ...

class OpExecutionError(Exception): ...

"""Service pool for distributed execution."""

from __future__ import annotations

import warnings
from typing import Any, Callable

from ..api.service import ServiceConfig
from .context import get_current_execution_context
from .utils import maybe_await
from ..resources import Mesh


class ServiceInstance:
    fn: Any = None
    state: Any = None

    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def shutdown(self):
        pass

    def get_service(self) -> Any | None:
        """Return the backing service instance for method calls."""
        if self.state is not None:
            return self.state
        return self.fn


class LocalServiceInstance(ServiceInstance):
    def __init__(self, fn: Callable, state: Any = None):
        self.fn = fn
        self.state = state

    async def execute(self, *args, **kwargs) -> Any:
        args = (self.state, *args) if self.state is not None else args
        return await maybe_await(self.fn, *args, **kwargs)


class RemoteServiceInstance(ServiceInstance):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._client = None

    async def execute(self, *args, **kwargs) -> Any:
        if self._client is None:
            from ..backends.grpc.client import GrpcClient

            self._client = GrpcClient(self.endpoint)
        return await self._client.execute(*args, **kwargs)

    async def shutdown(self):
        if self._client is not None:
            await self._client.close()


class ServicePool:
    def __init__(
        self,
        fn: Any,
        config: ServiceConfig,
        mesh: Mesh,
        *,
        create_local: Callable[[], ServiceInstance] | None = None,
        create_remote: Callable[[str], ServiceInstance] | None = None,
    ):
        self.fn = fn
        self.config = config
        self.mesh = mesh
        self.instances: list[ServiceInstance] = []
        self._index = 0
        self._endpoint_index = 0
        self._deployed = False
        self._create_local = create_local
        self._create_remote = create_remote

    def _next_endpoint(self) -> str:
        endpoints = self.mesh.get_endpoints()
        if not endpoints:
            raise RuntimeError("No endpoints available in mesh for remote deployment")
        endpoint = endpoints[self._endpoint_index % len(endpoints)]
        self._endpoint_index += 1
        return endpoint

    async def _create_instance(self) -> ServiceInstance:
        if self.config.backend == "grpc":
            endpoint = self._next_endpoint()
            if self._create_remote:
                return await maybe_await(self._create_remote, endpoint)
            return RemoteServiceInstance(endpoint)
        elif self.config.backend == "local":
            if self._create_local:
                return await maybe_await(self._create_local)
            state = await maybe_await(self.config.setup) if self.config.setup else None
            return LocalServiceInstance(self.fn, state)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    async def deploy(self, n: int = 1):
        if self._deployed:
            raise RuntimeError(f"Pool for {self.fn.__name__} already deployed")

        max_n = self.mesh.max_instances(self.config.gpus)
        if self.config.gpus and max_n == 0:
            warnings.warn(
                f"{self.fn.__name__} requires {self.config.gpus} GPUs "
                f"but mesh has {self.mesh.total_gpus} GPUs. Deploying single instance anyway.",
                RuntimeWarning,
                stacklevel=2,
            )
            n = 1
        elif n > max_n:
            raise ValueError(
                f"Cannot deploy {n} instances with {self.config.gpus} GPUs per instance. "
                f"Mesh has {self.mesh.total_gpus} GPUs total, max {max_n} instances."
            )

        for _ in range(n):
            self.instances.append(await self._create_instance())
        self._deployed = True

    def get_next(self) -> ServiceInstance:
        if not self._deployed or not self.instances:
            raise RuntimeError(f"Pool for {self.fn.__name__} not deployed")
        instance = self.instances[self._index]
        self._index = (self._index + 1) % len(self.instances)
        return instance

    async def shutdown(self):
        for instance in self.instances:
            await instance.shutdown()
        self.instances.clear()
        self._deployed = False

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    def __repr__(self):
        return f"ServicePool({self.fn.__name__}, {self.instance_count} instances, {'deployed' if self._deployed else 'not deployed'})"


def get_or_create_pool(fn: Any, config: ServiceConfig, mesh: Mesh) -> ServicePool:
    """Get existing pool or create new one.

    Args:
        fn: Function to create instances from
        config: Distribution configuration
        cluster: Cluster providing resources

    Returns:
        ServicePool for the function
    """
    pools = get_current_execution_context().pools
    pool_key = (id(mesh), id(fn))

    if pool_key not in pools:
        pool = ServicePool(fn, config, mesh)
        pools[pool_key] = pool

    return pools[pool_key]


def get_or_create_handle_pool(handle: Any, mesh: Mesh) -> ServicePool:
    """Get existing pool or create a new pool for a service handle.

    Args:
        handle: ServiceHandle to execute
        mesh: Mesh providing resources

    Returns:
        ServicePool for the handle
    """
    pools = get_current_execution_context().pools
    pool_key = (id(mesh), handle, "handle")

    if pool_key not in pools:

        def _create_local() -> ServiceInstance:
            instance = handle.service_class(*handle.init_args, **handle.init_kwargs)
            return LocalServiceInstance(instance, None)

        pool = ServicePool(
            fn=handle.service_class,
            config=handle.config,
            mesh=mesh,
            create_local=_create_local,
        )
        pools[pool_key] = pool

    return pools[pool_key]


async def ensure_deployed(pool: ServicePool):
    """Ensure pool is deployed (auto-deploy with n=1 if needed).

    Args:
        pool: Pool to check/deploy
    """
    if not pool._deployed:
        # Auto-deploy with single instance
        await pool.deploy(n=1)


def get_all_pools() -> dict[tuple[object, ...], ServicePool]:
    """Get all active pools (for debugging/monitoring).

    Returns:
        Dictionary of pool_key -> ServicePool
        Keys can be (mesh_id, fn_id) or (mesh_id, handle_hash, "handle")
    """
    return get_current_execution_context().pools.copy()


def _get_pools() -> dict[tuple[object, ...], "ServicePool"]:
    """Backwards-compatible access for tests."""
    return get_current_execution_context().pools


async def shutdown_all():
    """Shutdown all active pools."""
    pools = get_current_execution_context().pools
    for pool in list(pools.values()):
        await pool.shutdown()
    pools.clear()

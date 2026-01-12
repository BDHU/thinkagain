"""Replica pool for distributed execution."""

from __future__ import annotations

import contextvars
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable

from ...core.execution.replica import ReplicaConfig
from ...core.execution.runtime import maybe_await
from ..mesh import Mesh


# Context-local storage for active pools
_pools: contextvars.ContextVar[dict[tuple[object, ...], "ReplicaPool"] | None] = (
    contextvars.ContextVar("replica_pools", default=None)
)


def _get_pools() -> dict[tuple[object, ...], "ReplicaPool"]:
    """Get context-local pool registry keyed by pool-specific tuple.

    Keys can be:
    - (mesh_id, fn_id): 2-tuple for function pools
    - (mesh_id, handle_hash, "handle"): 3-tuple for replica handle pools
    """
    pools = _pools.get()
    if pools is None:
        pools = {}
        _pools.set(pools)
    return pools


class Replica(ABC):
    """Abstract base class for replica execution backends."""

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute function on this replica.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function execution
        """
        pass

    async def shutdown(self):
        """Shutdown this replica and cleanup resources."""
        pass


class LocalReplica(Replica):
    """Local replica that executes functions in the same process."""

    def __init__(self, fn: Callable, state: Any = None):
        """Initialize local replica.

        Args:
            fn: Function to execute
            state: Optional state from setup function
        """
        self.fn = fn
        self.state = state

    async def execute(self, *args, **kwargs) -> Any:
        """Execute function locally."""
        if self.state is not None:
            # If setup was provided, state is first argument
            return await maybe_await(self.fn, self.state, *args, **kwargs)
        else:
            # No setup, call function directly
            return await maybe_await(self.fn, *args, **kwargs)


class RemoteReplica(Replica):
    """Remote replica that executes functions via gRPC."""

    def __init__(self, endpoint: str):
        """Initialize remote replica.

        Args:
            endpoint: gRPC server endpoint (e.g., "server1:8000")
        """
        self.endpoint = endpoint
        self._client = None

    async def execute(self, *args, **kwargs) -> Any:
        """Execute function on remote server via gRPC."""
        if self._client is None:
            # Lazy initialization of gRPC client
            from ..grpc.client import GrpcClient

            self._client = GrpcClient(self.endpoint)

        return await self._client.execute(*args, **kwargs)

    async def shutdown(self):
        """Close gRPC connection."""
        if self._client is not None:
            await self._client.close()


class ReplicaPool:
    """Pool of replicas for load-balanced execution."""

    def __init__(
        self,
        fn: Any,  # Can be a function or replica class
        config: ReplicaConfig,
        mesh: Mesh,
        *,
        create_local: Callable[[], Replica] | None = None,
        create_remote: Callable[[str], Replica] | None = None,
    ):
        """Initialize replica pool.

        Args:
            fn: Function to replicate
            config: Distribution configuration
            mesh: Mesh providing resources
            create_local: Optional factory to create a local replica instance
            create_remote: Optional factory to create a remote replica instance
        """
        self.fn = fn
        self.config = config
        self.mesh = mesh
        self.replicas: list[Replica] = []
        self._current_index = 0
        self._endpoint_index = 0
        self._deployed = False
        self._create_local = create_local
        self._create_remote = create_remote

    def _get_next_endpoint(self) -> str:
        """Get next available endpoint from mesh (round-robin).

        Returns:
            Endpoint address for next replica

        Raises:
            RuntimeError: If no endpoints are available in the mesh
        """
        endpoints = self.mesh.get_endpoints()
        if not endpoints:
            raise RuntimeError(
                "No endpoints available in mesh for remote deployment. "
                "Please configure MeshNodes with endpoint addresses."
            )

        # Simple round-robin for now (can be made pluggable for optimizer)
        endpoint = endpoints[self._endpoint_index % len(endpoints)]
        self._endpoint_index += 1
        return endpoint

    async def _create_local_replica(self) -> Replica:
        if self._create_local is not None:
            return await maybe_await(self._create_local)
        if self.config.setup:
            state = await maybe_await(self.config.setup)
            return LocalReplica(self.fn, state)
        return LocalReplica(self.fn, None)

    async def _create_remote_replica(self) -> Replica:
        endpoint = self._get_next_endpoint()
        if self._create_remote is not None:
            return await maybe_await(self._create_remote, endpoint)
        return RemoteReplica(endpoint)

    async def _create_replica(self) -> Replica:
        creator = _BACKENDS.get(self.config.backend)
        if creator is None:
            raise ValueError(
                f"Unknown backend: {self.config.backend}. "
                f"Supported backends: {', '.join(sorted(_BACKENDS))}"
            )
        return await creator(self)

    async def deploy(self, n: int = 1):
        """Deploy n instances of the function.

        Args:
            n: Number of instances to deploy
        """
        if self._deployed:
            raise RuntimeError(f"Pool for {self.fn.__name__} already deployed")

        # Check mesh capacity
        max_n = self.mesh.max_instances(self.config.gpus)
        if self.config.gpus and max_n == 0:
            # GPU required but mesh has no GPUs - deploy single instance anyway
            # (will run on CPU, useful for development/testing)
            warnings.warn(
                f"{self.fn.__name__} requires {self.config.gpus} GPUs "
                f"but mesh has {self.mesh.total_gpus} GPUs. "
                "Deploying single instance anyway (will run on CPU).",
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
            replica = await self._create_replica()
            self.replicas.append(replica)

        self._deployed = True

    def get_next(self) -> Replica:
        """Get next replica (round-robin).

        Returns:
            Next replica to use

        Raises:
            RuntimeError: If pool not deployed
        """
        if not self._deployed or not self.replicas:
            raise RuntimeError(
                f"Pool for {self.fn.__name__} not deployed. "
                f"Call deploy() first or ensure auto-deploy is enabled."
            )

        replica = self.replicas[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.replicas)
        return replica

    async def shutdown(self):
        """Shutdown all replicas and cleanup resources."""
        for replica in self.replicas:
            await replica.shutdown()
        self.replicas.clear()
        self._deployed = False

    @property
    def instance_count(self) -> int:
        """Number of deployed instances."""
        return len(self.replicas)

    def __repr__(self):
        fn_name = self.fn.__name__
        status = "deployed" if self._deployed else "not deployed"
        return f"ReplicaPool({fn_name}, {self.instance_count} instances, {status})"


_BACKENDS: dict[str, Callable[[ReplicaPool], Any]] = {
    "local": ReplicaPool._create_local_replica,
    "grpc": ReplicaPool._create_remote_replica,
}


def get_or_create_pool(fn: Any, config: ReplicaConfig, mesh: Mesh) -> ReplicaPool:
    """Get existing pool or create new one.

    Args:
        fn: Function to replicate
        config: Distribution configuration
        cluster: Cluster providing resources

    Returns:
        ReplicaPool for the function
    """
    pools = _get_pools()
    pool_key = (id(mesh), id(fn))

    if pool_key not in pools:
        pool = ReplicaPool(fn, config, mesh)
        pools[pool_key] = pool

    return pools[pool_key]


def get_or_create_handle_pool(handle: Any, mesh: Mesh) -> ReplicaPool:
    """Get existing pool or create a new pool for a replica handle.

    Args:
        handle: ReplicaHandle to execute
        mesh: Mesh providing resources

    Returns:
        ReplicaPool for the handle
    """
    pools = _get_pools()
    pool_key = (id(mesh), hash(handle), "handle")

    if pool_key not in pools:

        def _create_local() -> Replica:
            instance = handle.replica_class(
                *handle.init_args, **dict(handle.init_kwargs)
            )
            return LocalReplica(instance, None)

        pool = ReplicaPool(
            fn=handle.replica_class,
            config=handle.config,
            mesh=mesh,
            create_local=_create_local,
        )
        pools[pool_key] = pool

    return pools[pool_key]


async def ensure_deployed(pool: ReplicaPool):
    """Ensure pool is deployed (auto-deploy with n=1 if needed).

    Args:
        pool: Pool to check/deploy
    """
    if not pool._deployed:
        # Auto-deploy with single instance
        await pool.deploy(n=1)


def get_all_pools() -> dict[tuple[object, ...], ReplicaPool]:
    """Get all active pools (for debugging/monitoring).

    Returns:
        Dictionary of pool_key -> ReplicaPool
        Keys can be (mesh_id, fn_id) or (mesh_id, handle_hash, "handle")
    """
    return _get_pools().copy()


async def shutdown_all():
    """Shutdown all active pools."""
    pools = _get_pools()
    for pool in list(pools.values()):
        await pool.shutdown()
    pools.clear()

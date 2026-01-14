"""Service pool for distributed execution."""

from __future__ import annotations

import time
import warnings
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from ..api.service import ServiceConfig
from .context import get_current_execution_context
from .metrics import sample_stats
from .utils import maybe_await
from ..resources import Mesh


# ---------------------------------------------------------------------------
# Lightweight Metrics (Non-blocking)
# ---------------------------------------------------------------------------


@dataclass
class ServiceMetrics:
    """Lightweight metrics for a service pool.

    Design principles:
    - No locks or synchronization in critical path
    - Simple atomic operations (int increments, time captures)
    - Heavy aggregation happens in background optimizer
    - Sampled latencies using fixed-size deque (no unbounded growth)
    """

    # Request counters (simple atomic increments)
    total_requests: int = 0
    total_errors: int = 0

    # Latency samples (fixed-size ring buffer, most recent N samples)
    # Using deque with maxlen for automatic eviction
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Concurrent request tracking
    current_concurrent: int = 0

    # Snapshot tracking for QPS (computed over deltas between snapshots)
    _last_snapshot_time: float = field(default_factory=time.perf_counter)
    _last_snapshot_total: int = 0

    @asynccontextmanager
    async def track(self):
        """Track a single request end-to-end with minimal overhead."""
        self.record_request_start()
        start_time = time.perf_counter()
        error = False
        try:
            yield
        except Exception:
            error = True
            raise
        finally:
            latency = time.perf_counter() - start_time
            self.record_request_end(latency, error=error)

    def record_request_start(self):
        """Record request start (called before execution).

        CRITICAL: This is in the serving path, must be extremely fast.
        Just increment a counter, no heavy computation.
        """
        self.current_concurrent += 1

    def record_request_end(self, latency_seconds: float, error: bool = False):
        """Record request completion (called after execution).

        CRITICAL: This is still close to serving path, keep it fast.
        - Simple increments/decrements
        - Append to fixed-size deque (O(1) with automatic eviction)
        - No percentile computation here (done in background)
        """
        self.current_concurrent -= 1
        self.total_requests += 1
        if error:
            self.total_errors += 1

        # Store latency sample (deque handles overflow automatically)
        self.latency_samples.append(latency_seconds)

    def get_snapshot(self) -> dict[str, Any]:
        """Get metrics snapshot for optimizer (called in background).

        This is NOT in the critical path. The optimizer calls this periodically
        to make scaling decisions. Heavy computation (percentiles) happens here.
        """
        current_time = time.perf_counter()
        elapsed = current_time - self._last_snapshot_time
        total_requests = self.total_requests
        delta_requests = total_requests - self._last_snapshot_total

        # Compute QPS (requests per second) over the last snapshot window.
        qps = delta_requests / elapsed if elapsed > 0 else 0.0

        # Compute latency stats from samples
        latency_stats = sample_stats(self.latency_samples)
        latency_p50 = latency_stats["p50"]
        latency_p95 = latency_stats["p95"]
        latency_p99 = latency_stats["p99"]
        latency_mean = latency_stats["mean"]

        error_rate = (
            self.total_errors / self.total_requests if self.total_requests > 0 else 0.0
        )

        snapshot = {
            "total_requests": total_requests,
            "total_errors": self.total_errors,
            "error_rate": error_rate,
            "qps": qps,
            "current_concurrent": self.current_concurrent,
            "latency_p50": latency_p50,
            "latency_p95": latency_p95,
            "latency_p99": latency_p99,
            "latency_mean": latency_mean,
            "sample_count": latency_stats["count"],
        }

        self._last_snapshot_time = current_time
        self._last_snapshot_total = total_requests

        return snapshot


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

        # Lightweight metrics (non-blocking)
        self.metrics = ServiceMetrics()

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

    async def execute_with_metrics(
        self, instance: ServiceInstance, *args, **kwargs
    ) -> Any:
        """Execute on instance with lightweight metrics tracking.

        CRITICAL PATH: This wraps the actual execution, so it must be extremely fast.
        - record_request_start(): just increments a counter (1 int increment)
        - record_request_end(): increments counters + appends to deque (O(1))
        - No locks, no heavy computation, no blocking

        The optimizer reads metrics via get_metrics_snapshot() in background.
        """
        async with self.metrics.track():
            return await instance.execute(*args, **kwargs)

    def get_metrics_snapshot(self) -> dict[str, Any]:
        """Get current metrics snapshot (called by optimizer in background).

        This is NOT in the critical path. Heavy computation (percentiles)
        happens here, not during serving.
        """
        snapshot = self.metrics.get_snapshot()
        snapshot["instance_count"] = self.instance_count
        snapshot["config"] = {
            "min_replicas": self.config.autoscaling.min_replicas,
            "max_replicas": self.config.autoscaling.max_replicas,
            "target_concurrent_requests": self.config.autoscaling.target_concurrent_requests,
        }
        return snapshot

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

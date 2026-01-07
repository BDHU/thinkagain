"""gRPC server for hosting replica instances."""

from __future__ import annotations

import asyncio
import inspect
import socket
from typing import Any

import grpc.aio

from .proto import replica_pb2, replica_pb2_grpc
from ..serialization import PickleSerializer, Serializer
from ..utils import RoundRobinPool, PoolBackendMixin


class ReplicaRegistry(PoolBackendMixin):
    """Server-side registry of replica classes and instances."""

    def __init__(self):
        self._classes: dict[str, type] = {}
        self._pool = RoundRobinPool()

    def register(self, cls: type) -> None:
        """Register a replica class."""
        self._classes[cls.__name__] = cls

    def deploy(
        self, name: str, instances: int, cpus: int, gpus: int, args: tuple, kwargs: dict
    ) -> None:
        """Deploy instances of a registered class.

        Args:
            name: Replica class name
            instances: Number of instances to deploy
            cpus: CPUs per instance (for resource tracking/scheduling)
            gpus: GPUs per instance (for resource tracking/scheduling)
            args: Constructor arguments
            kwargs: Constructor keyword arguments
        """
        if name not in self._classes:
            raise ValueError(f"Unknown replica class: {name}")
        cls = self._classes[name]
        # TODO: Use cpus/gpus for resource accounting and scheduling
        # For now, we simply deploy to pool
        _ = (cpus, gpus)  # Acknowledge parameters
        self._deploy_to_pool(name, cls, instances, args, kwargs)

    def shutdown(self, name: str) -> None:
        """Shutdown instances of a class."""
        self._shutdown_from_pool(name)

    def shutdown_all(self) -> None:
        """Shutdown all instances."""
        self._pool.clear()

    def get_instance(self, name: str) -> Any:
        """Get next instance using round-robin."""
        return self._get_from_pool(name)


class AsyncReplicaServicer(replica_pb2_grpc.ReplicaServiceServicer):
    """Async gRPC servicer for replica operations."""

    def __init__(self, registry: ReplicaRegistry, serializer: Serializer):
        self._registry = registry
        self._serializer = serializer

    async def Call(self, request, context):
        """Call a method on a replica instance (async)."""
        try:
            instance = self._registry.get_instance(request.replica_name)
            method = getattr(instance, request.method)
            args = self._serializer.loads(request.args) if request.args else ()
            kwargs = self._serializer.loads(request.kwargs) if request.kwargs else {}

            # Handle both async and sync methods
            if inspect.iscoroutinefunction(method):
                # Async method (e.g., vLLM async engine)
                result = await method(*args, **kwargs)
            else:
                # Sync method - run in thread pool to avoid blocking event loop
                result = await asyncio.to_thread(method, *args, **kwargs)

            return replica_pb2.CallResponse(result=self._serializer.dumps(result))
        except Exception as e:
            return replica_pb2.CallResponse(error=str(e))

    async def Deploy(self, request, context):
        """Deploy replica instances (async)."""
        try:
            args = self._serializer.loads(request.args) if request.args else ()
            kwargs = self._serializer.loads(request.kwargs) if request.kwargs else {}
            # Deployment is typically sync, run in thread pool
            await asyncio.to_thread(
                self._registry.deploy,
                request.replica_name,
                request.instances,
                request.cpus,
                request.gpus,
                args,
                kwargs,
            )
            return replica_pb2.DeployResponse(success=True)
        except Exception as e:
            return replica_pb2.DeployResponse(success=False, error=str(e))

    async def Shutdown(self, request, context):
        """Shutdown replica instances (async)."""
        try:
            if request.replica_name:
                await asyncio.to_thread(self._registry.shutdown, request.replica_name)
            else:
                await asyncio.to_thread(self._registry.shutdown_all)
            return replica_pb2.ShutdownResponse(success=True)
        except Exception as e:
            return replica_pb2.ShutdownResponse(success=False, error=str(e))


async def serve(
    port: int = 50051,
    registry: ReplicaRegistry | None = None,
    *,
    serializer: Serializer | None = None,
    bind_host: str = "0.0.0.0",
) -> tuple[grpc.aio.Server, int]:
    """Start the async gRPC server.

    Args:
        port: Port to listen on.
        registry: Optional pre-configured registry. If None, creates empty one.
        serializer: Optional serializer for request/response payloads.

    Returns:
        A tuple of (grpc.aio.Server, bound_port).
    """
    if registry is None:
        registry = ReplicaRegistry()
    if serializer is None:
        serializer = PickleSerializer()

    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, 0))
            port = sock.getsockname()[1]

    server = grpc.aio.server()
    replica_pb2_grpc.add_ReplicaServiceServicer_to_server(
        AsyncReplicaServicer(registry, serializer), server
    )
    bound_port = server.add_insecure_port(f"{bind_host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind gRPC server to port {port}")
    await server.start()
    return server, bound_port


async def run_server(
    port: int = 50051,
    registry: ReplicaRegistry | None = None,
    *,
    serializer: Serializer | None = None,
    bind_host: str = "0.0.0.0",
) -> None:
    """Run the async gRPC server.

    Args:
        port: Port to listen on.
        registry: Optional pre-configured registry.
        serializer: Optional serializer for request/response payloads.

    Example:
        asyncio.run(run_server(port=50051))
    """
    server, bound_port = await serve(
        port,
        registry,
        serializer=serializer,
        bind_host=bind_host,
    )
    print(f"thinkagain gRPC server listening on port {bound_port}")
    await server.wait_for_termination()

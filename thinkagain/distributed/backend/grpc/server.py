"""gRPC server for hosting replica instances."""

from __future__ import annotations

import pickle
from concurrent import futures
from typing import Any

import grpc

from .proto import replica_pb2, replica_pb2_grpc


class ReplicaRegistry:
    """Server-side registry of replica classes and instances."""

    def __init__(self):
        self._classes: dict[str, type] = {}
        self._instances: dict[str, list] = {}
        self._round_robin_idx: dict[str, int] = {}

    def register(self, cls: type) -> None:
        """Register a replica class."""
        self._classes[cls.__name__] = cls

    def deploy(self, name: str, n: int, args: tuple, kwargs: dict) -> None:
        """Deploy n instances of a registered class."""
        if name not in self._classes:
            raise ValueError(f"Unknown replica class: {name}")
        cls = self._classes[name]

        initializer = getattr(cls, "__local_init__", cls)
        self._instances[name] = [initializer(*args, **kwargs) for _ in range(n)]
        self._round_robin_idx[name] = 0

    def shutdown(self, name: str) -> None:
        """Shutdown instances of a class."""
        if name in self._instances:
            del self._instances[name]
            del self._round_robin_idx[name]

    def shutdown_all(self) -> None:
        """Shutdown all instances."""
        self._instances.clear()
        self._round_robin_idx.clear()

    def get_instance(self, name: str) -> Any:
        """Get next instance using round-robin."""
        if name not in self._instances or not self._instances[name]:
            raise RuntimeError(f"No instances for replica '{name}'")
        instances = self._instances[name]
        idx = self._round_robin_idx[name]
        self._round_robin_idx[name] = (idx + 1) % len(instances)
        return instances[idx]


class ReplicaServicer(replica_pb2_grpc.ReplicaServiceServicer):
    """gRPC servicer for replica operations."""

    def __init__(self, registry: ReplicaRegistry):
        self._registry = registry

    def Call(self, request, context):
        """Call a method on a replica instance."""
        try:
            instance = self._registry.get_instance(request.replica_name)
            method = getattr(instance, request.method)
            args = pickle.loads(request.args) if request.args else ()
            kwargs = pickle.loads(request.kwargs) if request.kwargs else {}
            result = method(*args, **kwargs)
            return replica_pb2.CallResponse(result=pickle.dumps(result))
        except Exception as e:
            return replica_pb2.CallResponse(error=str(e))

    def Deploy(self, request, context):
        """Deploy replica instances."""
        try:
            args = pickle.loads(request.args) if request.args else ()
            kwargs = pickle.loads(request.kwargs) if request.kwargs else {}
            self._registry.deploy(request.replica_name, request.n, args, kwargs)
            return replica_pb2.DeployResponse(success=True)
        except Exception as e:
            return replica_pb2.DeployResponse(success=False, error=str(e))

    def Shutdown(self, request, context):
        """Shutdown replica instances."""
        try:
            if request.replica_name:
                self._registry.shutdown(request.replica_name)
            else:
                self._registry.shutdown_all()
            return replica_pb2.ShutdownResponse(success=True)
        except Exception as e:
            return replica_pb2.ShutdownResponse(success=False, error=str(e))


def serve(
    port: int = 50051, registry: ReplicaRegistry | None = None
) -> tuple[grpc.Server, int]:
    """Start the gRPC server.

    Args:
        port: Port to listen on.
        registry: Optional pre-configured registry. If None, creates empty one.

    Returns:
        A tuple of (grpc.Server, bound_port).
    """
    if registry is None:
        registry = ReplicaRegistry()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    replica_pb2_grpc.add_ReplicaServiceServicer_to_server(
        ReplicaServicer(registry), server
    )
    bound_port = server.add_insecure_port(f"0.0.0.0:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind gRPC server to port {port}")
    server.start()
    return server, bound_port


def run_server(port: int = 50051, registry: ReplicaRegistry | None = None) -> None:
    """Run the gRPC server (blocking).

    Args:
        port: Port to listen on.
        registry: Optional pre-configured registry.
    """
    server, bound_port = serve(port, registry)
    print(f"thinkagain gRPC server listening on port {bound_port}")
    server.wait_for_termination()

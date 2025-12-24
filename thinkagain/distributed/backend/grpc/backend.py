"""gRPC backend for distributed replica execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thinkagain.distributed.replica import ReplicaSpec

import functools

from ..serialization import PickleSerializer, Serializer


@functools.cache
def _get_grpc():
    """Lazily import grpc and generated stubs."""
    import grpc

    from .proto import replica_pb2, replica_pb2_grpc

    return grpc, replica_pb2, replica_pb2_grpc


class GrpcReplicaProxy:
    """Proxy that routes method calls to remote replica via gRPC."""

    def __init__(self, stub, replica_name: str, serializer: Serializer):
        object.__setattr__(self, "_stub", stub)
        object.__setattr__(self, "_replica_name", replica_name)
        object.__setattr__(self, "_serializer", serializer)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            _, replica_pb2, _ = _get_grpc()
            request = replica_pb2.CallRequest(
                replica_name=self._replica_name,
                method=name,
                args=self._serializer.dumps(args),
                kwargs=self._serializer.dumps(kwargs),
            )
            response = self._stub.Call(request)
            if response.error:
                raise RuntimeError(f"Remote call failed: {response.error}")
            return self._serializer.loads(response.result)

        return call


class GrpcBackend:
    """gRPC backend for remote replica execution."""

    def __init__(
        self,
        address: str | None,
        options: dict | None = None,
        *,
        serializer: Serializer | None = None,
    ):
        if not address:
            raise ValueError(
                "GrpcBackend requires an address (e.g., 'localhost:50051')"
            )
        self._address = address
        self._options = options or {}
        self._channel = None
        self._stub = None
        self._deployed: set[str] = set()
        self._serializer = serializer or PickleSerializer()

    def _ensure_connected(self):
        """Lazily connect to the gRPC server."""
        if self._channel is None:
            grpc, _, replica_pb2_grpc = _get_grpc()
            self._channel = grpc.insecure_channel(self._address)
            self._stub = replica_pb2_grpc.ReplicaServiceStub(self._channel)

    def deploy(self, spec: "ReplicaSpec", *args, **kwargs) -> None:
        """Request the server to deploy instances."""
        name = spec.cls.__name__
        if name in self._deployed:
            return
        self._ensure_connected()
        _, replica_pb2, _ = _get_grpc()
        request = replica_pb2.DeployRequest(
            replica_name=name,
            n=spec.n,
            args=self._serializer.dumps(args),
            kwargs=self._serializer.dumps(kwargs),
        )
        response = self._stub.Deploy(request)
        if not response.success:
            raise RuntimeError(f"Remote deploy failed: {response.error}")
        self._deployed.add(name)

    def shutdown(self, spec: "ReplicaSpec") -> None:
        """Request the server to shutdown instances."""
        name = spec.cls.__name__
        if name not in self._deployed:
            return
        self._ensure_connected()
        _, replica_pb2, _ = _get_grpc()
        request = replica_pb2.ShutdownRequest(replica_name=name)
        response = self._stub.Shutdown(request)
        if not response.success:
            raise RuntimeError(f"Remote shutdown failed: {response.error}")
        self._deployed.discard(name)

    def get_instance(self, spec: "ReplicaSpec") -> GrpcReplicaProxy:
        """Return a proxy that routes calls to remote instances."""
        self._ensure_connected()
        return GrpcReplicaProxy(self._stub, spec.cls.__name__, self._serializer)

    def is_deployed(self, spec: "ReplicaSpec") -> bool:
        return spec.cls.__name__ in self._deployed

    def close(self):
        """Close the gRPC channel."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
        self._deployed.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

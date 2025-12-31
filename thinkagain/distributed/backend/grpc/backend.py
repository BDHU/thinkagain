"""gRPC backend for distributed replica execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thinkagain.distributed.replica import ReplicaSpec

from ..serialization import PickleSerializer, Serializer

# Lazy imports for grpc to avoid dependency when not used
grpc_aio = None
replica_pb2 = None
replica_pb2_grpc = None


def _ensure_grpc():
    """Lazily import grpc and generated stubs."""
    global grpc_aio, replica_pb2, replica_pb2_grpc
    if grpc_aio is None:
        import grpc.aio as _grpc_aio
        from .proto import replica_pb2 as _replica_pb2
        from .proto import replica_pb2_grpc as _replica_pb2_grpc

        grpc_aio = _grpc_aio
        replica_pb2 = _replica_pb2
        replica_pb2_grpc = _replica_pb2_grpc


class GrpcReplicaProxy:
    """Sync proxy that routes method calls to remote replica via gRPC.

    Only methods are exposed (not attributes). Methods return async callables
    that perform the actual RPC.
    """

    def __init__(self, backend: "AsyncGrpcBackend", replica_name: str):
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_replica_name", replica_name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        # Return an async callable that performs the RPC
        async def call(*args, **kwargs):
            await self._backend._ensure_connected()
            _ensure_grpc()
            request = replica_pb2.CallRequest(
                replica_name=self._replica_name,
                method=name,
                args=self._backend._serializer.dumps(args),
                kwargs=self._backend._serializer.dumps(kwargs),
            )
            response = await self._backend._stub.Call(request)
            if response.error:
                raise RuntimeError(f"Remote call failed: {response.error}")
            return self._backend._serializer.loads(response.result)

        return call


class AsyncGrpcBackend:
    """gRPC backend for remote replica execution (sync interface, async I/O)."""

    def __init__(
        self,
        address: str | None,
        options: dict | None = None,
        *,
        serializer: Serializer | None = None,
    ):
        if not address:
            raise ValueError(
                "AsyncGrpcBackend requires an address (e.g., 'localhost:50051')"
            )
        self._address = address
        self._options = options or {}
        self._channel = None
        self._stub = None
        self._deployed: set[str] = set()
        self._serializer = serializer or PickleSerializer()

    async def _ensure_connected(self):
        """Lazily connect to the async gRPC server."""
        if self._channel is None:
            _ensure_grpc()
            self._channel = grpc_aio.insecure_channel(self._address)
            self._stub = replica_pb2_grpc.ReplicaServiceStub(self._channel)

    async def deploy(self, spec: "ReplicaSpec", *args, **kwargs) -> None:
        """Request the server to deploy instances (async)."""
        name = spec.name
        if name in self._deployed:
            return
        await self._ensure_connected()
        request = replica_pb2.DeployRequest(
            replica_name=name,
            n=spec.n,
            args=self._serializer.dumps(args),
            kwargs=self._serializer.dumps(kwargs),
        )
        response = await self._stub.Deploy(request)
        if not response.success:
            raise RuntimeError(f"Remote deploy failed: {response.error}")
        self._deployed.add(name)

    async def shutdown(self, spec: "ReplicaSpec") -> None:
        """Request the server to shutdown instances (async)."""
        name = spec.name
        if name not in self._deployed:
            return
        await self._ensure_connected()
        request = replica_pb2.ShutdownRequest(replica_name=name)
        response = await self._stub.Shutdown(request)
        if not response.success:
            raise RuntimeError(f"Remote shutdown failed: {response.error}")
        self._deployed.discard(name)

    def get_instance(self, spec: "ReplicaSpec") -> GrpcReplicaProxy:
        """Return a proxy that routes calls to remote instances (immediate return).

        The proxy will handle connection lazily when methods are called.
        """
        name = spec.name
        return GrpcReplicaProxy(self, name)

    def is_deployed(self, spec: "ReplicaSpec") -> bool:
        return spec.name in self._deployed

    async def close(self):
        """Close the gRPC channel (async)."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
        self._deployed.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

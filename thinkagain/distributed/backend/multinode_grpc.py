"""Multi-node gRPC backend for distributed replica execution.

This backend spawns one gRPC server process per instance, enabling:
- True multi-node deployment
- Resource-based scheduling (cpus/gpus per instance)
- Dynamic scaling based on optimizer recommendations

Architecture:
    @replica(cpus=1) DemoClass
    await DemoClass.deploy(instances=2)
    → Spawns 2 server processes (potentially on different machines)
    → Each process hosts 1 DemoClass instance
    → Client round-robins between the 2 servers
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thinkagain.distributed.replica import ReplicaSpec

from ..backend.serialization import PickleSerializer, Serializer
from ..backend.utils import RoundRobinPool

# Lazy imports for grpc
grpc_aio = None
replica_pb2 = None
replica_pb2_grpc = None


def _ensure_grpc():
    """Lazily import grpc and generated stubs."""
    global grpc_aio, replica_pb2, replica_pb2_grpc
    if grpc_aio is None:
        from thinkagain.distributed.backend.grpc.proto import (
            replica_pb2 as _replica_pb2,
        )
        from thinkagain.distributed.backend.grpc.proto import (
            replica_pb2_grpc as _replica_pb2_grpc,
        )

        import grpc.aio as _grpc_aio

        grpc_aio = _grpc_aio
        replica_pb2 = _replica_pb2
        replica_pb2_grpc = _replica_pb2_grpc


@dataclass
class ServerNode:
    """Represents a running gRPC server process."""

    process: subprocess.Popen
    port: int
    replica_name: str
    cpus: int
    gpus: int
    channel: Any = None  # grpc.aio.Channel
    stub: Any = None  # ReplicaServiceStub


class MultiNodeGrpcBackend:
    """Multi-node gRPC backend that spawns one server per instance.

    Key features:
    - Spawns server processes dynamically during deploy()
    - Each instance gets its own server process
    - Maintains connection pool to all servers
    - Round-robins across servers for load balancing
    """

    def __init__(
        self,
        options: dict | None = None,
        *,
        serializer: Serializer | None = None,
    ):
        self._options = options or {}
        self._serializer = serializer or PickleSerializer()
        # replica_name -> list of ServerNode
        self._servers: dict[str, list[ServerNode]] = {}
        self._pool = RoundRobinPool()

    async def deploy(
        self, spec: "ReplicaSpec", instances: int = 1, *args, **kwargs
    ) -> None:
        """Deploy instances by spawning one gRPC server per instance.

        Supports incremental scaling:
        - If not deployed: spawn 'instances' servers
        - If already deployed with N servers:
          - instances > N: spawn (instances - N) more servers
          - instances < N: shutdown (N - instances) servers
          - instances == N: no-op

        Args:
            spec: Replica specification
            instances: Desired total number of instances
            *args: Constructor arguments for replica instances
            **kwargs: Constructor keyword arguments for replica instances
        """
        name = spec.name
        current_count = len(self._servers.get(name, []))

        # Case 1: Already at desired count
        if current_count == instances:
            return

        # Case 2: Scale down
        if current_count > instances:
            await self._scale_down(spec, instances)
            return

        # Case 3: Scale up (or initial deployment)
        await self._scale_up(spec, instances, current_count, args, kwargs)

    async def _scale_up(
        self,
        spec: "ReplicaSpec",
        desired_count: int,
        current_count: int,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Scale up by spawning additional servers."""
        name = spec.name

        # Get or create server list
        if name not in self._servers:
            self._servers[name] = []

        servers = self._servers[name]

        # Create server payload
        server_payload = self._create_server_payload(spec, args, kwargs)

        # Spawn additional servers
        new_servers = []
        instances_to_add = desired_count - current_count

        for i in range(instances_to_add):
            instance_id = current_count + i
            try:
                server_node = await self._spawn_server(server_payload, spec)
                new_servers.append(server_node)
            except Exception as e:
                # Cleanup newly spawned servers on failure
                for s in new_servers:
                    s.process.terminate()
                raise RuntimeError(
                    f"Failed to spawn server {instance_id} during scale-up: {e}"
                ) from e

        # Add new servers to the pool
        servers.extend(new_servers)
        self._pool.set_pool(name, servers, keep_index=True)

    async def _scale_down(self, spec: "ReplicaSpec", desired_count: int) -> None:
        """Scale down by shutting down excess servers."""
        name = spec.name
        servers = self._servers[name]
        current_count = len(servers)

        instances_to_remove = current_count - desired_count

        # Shutdown the last N servers
        servers_to_shutdown = servers[-instances_to_remove:]
        servers_to_keep = servers[:-instances_to_remove]

        for server in servers_to_shutdown:
            await self._shutdown_server(server, name)

        if servers_to_keep:
            self._servers[name] = servers_to_keep
            self._pool.set_pool(name, servers_to_keep, keep_index=True)
        else:
            del self._servers[name]
            self._pool.remove_pool(name)

    def _create_server_payload(
        self, spec: "ReplicaSpec", args: tuple, kwargs: dict
    ) -> bytes:
        """Serialize payload for a server process."""
        import cloudpickle

        payload = {
            "cls": spec.cls,
            "args": args,
            "kwargs": kwargs,
            "cpus": spec.cpus,
            "gpus": spec.gpus,
        }
        return cloudpickle.dumps(payload)

    async def _spawn_server(
        self, server_payload: bytes, spec: "ReplicaSpec"
    ) -> ServerNode:
        """Spawn a single server process and wait for it to be ready."""
        import socket
        import time

        # Write payload to temp file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            f.write(server_payload)
            payload_path = f.name

        try:
            # Start server process on random port
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "thinkagain.distributed.backend.grpc.server_main",
                    payload_path,
                    "0",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Wait for READY signal
            port = None
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                line = process.stdout.readline()
                if line.startswith("READY:"):
                    port = int(line.split(":")[1].strip())
                    break
                if process.poll() is not None:
                    stderr = process.stderr.read()
                    raise RuntimeError(f"Server died during startup: {stderr}")
                await asyncio.sleep(0.05)

            if port is None:
                process.terminate()
                raise RuntimeError("Server failed to start within timeout")

            # Verify server is listening
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("localhost", port), timeout=0.1):
                        break
                except OSError:
                    await asyncio.sleep(0.05)
            else:
                process.terminate()
                raise RuntimeError(f"Cannot connect to server on port {port}")

            # Create gRPC channel
            _ensure_grpc()
            channel = grpc_aio.insecure_channel(f"localhost:{port}")
            stub = replica_pb2_grpc.ReplicaServiceStub(channel)

            return ServerNode(
                process=process,
                port=port,
                replica_name=spec.name,
                cpus=spec.cpus,
                gpus=spec.gpus,
                channel=channel,
                stub=stub,
            )

        finally:
            # Cleanup temp file
            Path(payload_path).unlink(missing_ok=True)

    async def shutdown(self, spec: "ReplicaSpec") -> None:
        """Shutdown all server processes for this replica."""
        name = spec.name
        if name not in self._servers:
            return

        servers = self._servers[name]
        for server in servers:
            await self._shutdown_server(server, name)

        del self._servers[name]
        self._pool.remove_pool(name)

    async def _shutdown_server(self, server: ServerNode, replica_name: str) -> None:
        """Shutdown a single server process."""
        try:
            if server.stub:
                _ensure_grpc()
                request = replica_pb2.ShutdownRequest(replica_name=replica_name)
                await server.stub.Shutdown(request)
        except Exception:
            pass  # Ignore errors during shutdown

        if server.channel:
            await server.channel.close()

        server.process.terminate()
        try:
            server.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait()

    def get_instance(self, spec: "ReplicaSpec") -> Any:
        """Get proxy to next server instance using round-robin."""
        name = spec.name
        if not self._pool.has_pool(name):
            raise RuntimeError(f"Replica '{name}' not deployed")

        # Return proxy to this specific server
        # We need a modified proxy that uses a specific stub
        server = self._pool.get_next(name)
        return _SingleServerProxy(server, name, self._serializer)

    def is_deployed(self, spec: "ReplicaSpec") -> bool:
        """Check if replica is deployed."""
        return self._pool.has_pool(spec.name)

    async def close(self):
        """Close all server processes and connections."""
        for name in list(self._servers.keys()):
            # Create a temporary spec just for shutdown
            from thinkagain.distributed.replica import ReplicaSpec

            temp_spec = ReplicaSpec(cls=type(name, (), {}), cpus=1)
            await self.shutdown(temp_spec)


class _SingleServerProxy:
    """Proxy that routes to a specific server node."""

    def __init__(self, server: ServerNode, replica_name: str, serializer: Serializer):
        object.__setattr__(self, "_server", server)
        object.__setattr__(self, "_replica_name", replica_name)
        object.__setattr__(self, "_serializer", serializer)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        async def call(*args, **kwargs):
            _ensure_grpc()
            request = replica_pb2.CallRequest(
                replica_name=self._replica_name,
                method=name,
                args=self._serializer.dumps(args),
                kwargs=self._serializer.dumps(kwargs),
            )
            response = await self._server.stub.Call(request)
            if response.error:
                raise RuntimeError(f"Remote call failed: {response.error}")
            return self._serializer.loads(response.result)

        return call

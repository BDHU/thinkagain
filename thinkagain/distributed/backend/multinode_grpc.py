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

import os
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
from ..nodes import NodeConfig, NodeState
from ..scheduler import PlacementScheduler
from ..spawner import SpawnedProcess, create_spawner

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

    process: SpawnedProcess
    host: str  # hostname/IP where server is running
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

        # Parse node configuration from options
        node_configs = self._options.get("nodes")
        if node_configs is None:
            # Default: localhost with actual CPU count
            cpu_count = os.cpu_count() or 1
            node_configs = [NodeConfig(host="localhost", cpus=cpu_count, gpus=0)]
        elif isinstance(node_configs, list) and node_configs:
            # Ensure we have NodeConfig objects
            node_configs = [
                nc if isinstance(nc, NodeConfig) else NodeConfig(**nc)
                for nc in node_configs
            ]

        # Initialize node registry
        self.nodes = {nc.host: NodeState.from_config(nc) for nc in node_configs}
        self._node_configs = {nc.host: nc for nc in node_configs}

        # Initialize placement scheduler
        self.scheduler = PlacementScheduler(self.nodes)

        # Initialize process spawner
        self.spawner = create_spawner(self._node_configs)

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
        instances_to_add = desired_count - current_count

        # Get or create server list
        if name not in self._servers:
            self._servers[name] = []

        # Ask scheduler where to place new instances
        try:
            target_hosts = self.scheduler.select_nodes(
                replica_name=name,
                count=instances_to_add,
                cpus_per_instance=spec.cpus,
                gpus_per_instance=spec.gpus,
            )
        except RuntimeError as e:
            raise RuntimeError(f"Placement failed: {e}") from e

        # Create server payload
        server_payload = self._create_server_payload(spec, args, kwargs)

        # Spawn servers on selected nodes
        new_servers = []
        for target_host in target_hosts:
            try:
                server_node = await self._spawn_server(
                    server_payload, spec, target_host
                )
                new_servers.append(server_node)
            except Exception as e:
                # Cleanup newly spawned servers on failure
                for s in new_servers:
                    s.process.terminate()
                # Release reserved resources for all placements
                for placed_host in target_hosts:
                    self.nodes[placed_host].release(spec.cpus, spec.gpus)
                raise RuntimeError(
                    f"Failed to spawn server on {target_host}: {e}"
                ) from e

        # Add new servers to the pool
        self._servers[name].extend(new_servers)
        self._pool.set_pool(name, self._servers[name], keep_index=True)

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
            # Release resources back to node
            self.nodes[server.host].release(spec.cpus, spec.gpus)

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
        self, server_payload: bytes, spec: "ReplicaSpec", target_host: str
    ) -> ServerNode:
        """Spawn a single server process and wait for it to be ready.

        Args:
            server_payload: Serialized server configuration
            spec: Replica specification
            target_host: Hostname where server should be spawned
        """
        import socket
        import time

        # Write payload to temp file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            f.write(server_payload)
            payload_path = f.name

        try:
            # Build command
            command = [
                sys.executable,
                "-m",
                "thinkagain.distributed.backend.grpc.server_main",
                payload_path,
                "0",  # port 0 = random port
            ]

            # Get node config
            node_config = self._node_configs[target_host]

            # Spawn server process (local or remote)
            spawned = await self.spawner.spawn(
                node=node_config,
                command=command,
                payload_path=payload_path,
            )

            # Verify server is listening
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(
                        (spawned.host, spawned.port), timeout=0.1
                    ):
                        break
                except OSError:
                    await asyncio.sleep(0.05)
            else:
                spawned.terminate()
                raise RuntimeError(
                    f"Cannot connect to server on {spawned.host}:{spawned.port}"
                )

            # Create gRPC channel
            _ensure_grpc()
            channel = grpc_aio.insecure_channel(f"{spawned.host}:{spawned.port}")
            stub = replica_pb2_grpc.ReplicaServiceStub(channel)

            return ServerNode(
                process=spawned,
                host=spawned.host,
                port=spawned.port,
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

    def get_cluster_state(self) -> dict[str, dict]:
        """Get current cluster resource state.

        Returns:
            Mapping of hostname -> resource info with utilization
        """
        return self.scheduler.get_cluster_state()

    async def close(self):
        """Close all server processes and connections."""
        for name in list(self._servers.keys()):
            # Create a temporary spec just for shutdown
            from thinkagain.distributed.replica import ReplicaSpec

            temp_spec = ReplicaSpec(cls=type(name, (), {}), cpus=1)
            await self.shutdown(temp_spec)

        # Cleanup spawner resources (SSH connections, etc.)
        await self.spawner.cleanup()


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

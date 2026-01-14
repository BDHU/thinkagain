"""gRPC client for remote replica execution."""

from __future__ import annotations

import cloudpickle as pickle
from typing import Any

import grpc

from .proto import replica_pb2, replica_pb2_grpc


class GrpcClient:
    """Client for executing functions on remote gRPC servers."""

    def __init__(self, endpoint: str):
        """Initialize gRPC client.

        Args:
            endpoint: Server address (e.g., "localhost:8000" or "server1:8000")
        """
        self.endpoint = endpoint
        self.channel = grpc.aio.insecure_channel(endpoint)  # type: ignore[attr-defined]
        self.stub = replica_pb2_grpc.ReplicaServiceStub(self.channel)

    async def execute(self, *args, **kwargs) -> Any:
        """Execute function on remote server.

        Args:
            *args: Positional arguments to pass to the remote function
            **kwargs: Keyword arguments to pass to the remote function

        Returns:
            Result from remote function execution

        Raises:
            grpc.RpcError: If the RPC fails
            Exception: If the remote function raises an exception
        """
        # Serialize arguments using pickle
        args_bytes = pickle.dumps((args, kwargs))

        # Create request
        request = replica_pb2.ExecuteRequest(args=args_bytes)

        # Execute RPC
        response = await self.stub.Execute(request)

        # Check for errors
        if response.error:
            raise RuntimeError(f"Remote execution failed: {response.error}")

        # Deserialize and return result
        result = pickle.loads(response.result)
        return result

    async def close(self):
        """Close the gRPC channel."""
        await self.channel.close()

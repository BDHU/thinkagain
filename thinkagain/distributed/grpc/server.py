"""gRPC server for hosting replicated functions."""

from __future__ import annotations

import cloudpickle as pickle
import traceback
from typing import Any

import grpc

from .proto import replica_pb2, replica_pb2_grpc


class ReplicaServicer(replica_pb2_grpc.ReplicaServiceServicer):
    """gRPC servicer that hosts a replicated function instance."""

    def __init__(self, instance: Any):
        """Initialize servicer with a function instance.

        Args:
            instance: The instantiated replica (e.g., LLMServer instance)
                     Must have an async __call__ method
        """
        self.instance = instance

        # Verify instance is callable
        if not callable(instance):
            raise ValueError(
                f"Instance must be callable (have __call__ method), "
                f"got {type(instance).__name__}"
            )

    async def Execute(
        self, request: replica_pb2.ExecuteRequest, context: grpc.aio.ServicerContext
    ) -> replica_pb2.ExecuteResponse:
        """Execute the function with provided arguments.

        Args:
            request: ExecuteRequest containing pickled arguments
            context: gRPC context

        Returns:
            ExecuteResponse containing pickled result or error message
        """
        try:
            # Deserialize arguments
            args, kwargs = pickle.loads(request.args)

            # Execute function
            result = await self.instance(*args, **kwargs)

            # Serialize result
            result_bytes = pickle.dumps(result)

            return replica_pb2.ExecuteResponse(result=result_bytes, error="")

        except Exception as e:
            # Capture error details
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return replica_pb2.ExecuteResponse(result=b"", error=error_msg)


async def serve(instance: Any, port: int = 8000):
    """Start a gRPC server hosting the replica instance.

    Args:
        instance: The instantiated replica to host
        port: Port to listen on (default: 8000)

    Example:
        >>> @replica(gpus=1, backend="grpc")
        >>> class LLMServer:
        >>>     def __init__(self):
        >>>         self.model = load_model()
        >>>     async def __call__(self, prompt):
        >>>         return self.model.generate(prompt)
        >>>
        >>> instance = LLMServer()
        >>> await serve(instance, port=8000)
    """
    server = grpc.aio.server()
    replica_pb2_grpc.add_ReplicaServiceServicer_to_server(
        ReplicaServicer(instance), server
    )

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    print(f"Starting replica server on {listen_addr}")
    print(f"Serving: {type(instance).__name__}")

    await server.start()
    await server.wait_for_termination()

"""Entrypoint for multi-node gRPC server processes."""

from __future__ import annotations

import asyncio
import os
import sys


async def main() -> None:
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: server_main.py <payload_path> <port> [bind_host]",
            file=sys.stderr,
        )
        sys.exit(2)

    payload_path = sys.argv[1]
    port = int(sys.argv[2])
    bind_host = sys.argv[3] if len(sys.argv) == 4 else "0.0.0.0"

    try:
        import cloudpickle

        from thinkagain.distributed.backend.grpc import ReplicaRegistry, serve

        with open(payload_path, "rb") as f:
            payload = cloudpickle.loads(f.read())

        cls = payload["cls"]
        args = payload["args"]
        kwargs = payload["kwargs"]
        cpus = payload["cpus"]
        gpus = payload["gpus"]

        registry = ReplicaRegistry()
        registry.register(cls)

        server, bound_port = await serve(
            port=port,
            registry=registry,
            bind_host=bind_host,
        )

        registry.deploy(
            name=cls.__name__,
            instances=1,
            cpus=cpus,
            gpus=gpus,
            args=args,
            kwargs=kwargs,
        )

        print(f"READY:{bound_port}:{os.getpid()}", flush=True)
        await server.wait_for_termination()
    except Exception as exc:
        print(f"ERROR:{exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

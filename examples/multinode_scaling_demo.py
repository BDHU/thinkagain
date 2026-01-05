"""Demo of multi-node deployment with automatic incremental scaling.

This example demonstrates:
1. @replica(cpus=1) DemoClass - defines resource requirement per instance
2. deploy(instances=2) - spawns 2 gRPC server processes
3. deploy(instances=3) - spawns 1 more server (scale up)
4. deploy(instances=1) - shuts down 2 servers (scale down)

Each instance runs in its own gRPC server process.
"""

import asyncio

from thinkagain.distributed import init, replica


@replica(cpus=1)
class DemoClass:
    """Demo service - requires 1 CPU per instance."""

    def __init__(self, prefix: str = "demo"):
        self.prefix = prefix
        self.call_count = 0

    def process(self, data: str) -> str:
        self.call_count += 1
        return f"{self.prefix}:{data}:count={self.call_count}"

    def get_count(self) -> int:
        return self.call_count


async def main():
    print("=" * 60)
    print("Multi-Node Incremental Scaling Demo")
    print("=" * 60)

    try:
        # Initialize multi-node gRPC backend
        init(backend="grpc")

        # Step 1: Initial deployment with 2 instances
        print("\n[Step 1] Initial deployment: 2 instances")
        print("  → Spawning 2 gRPC server processes...")
        await DemoClass.deploy(instances=2, prefix="node")
        print("  ✓ 2 server processes running\n")

        # Test round-robin
        print("Testing round-robin across 2 servers:")
        for i in range(4):
            instance = DemoClass.get()
            result = await instance.process(f"data{i}")
            print(f"  Call {i + 1}: {result}")

        # Step 2: Scale up to 3 instances
        print("\n[Step 2] Scale up to 3 instances")
        print("  → Spawning 1 additional server process...")
        await DemoClass.deploy(instances=3, prefix="node")
        print("  ✓ 3 server processes running (1 new, 2 existing)\n")

        # Test round-robin with 3 servers
        print("Testing round-robin across 3 servers:")
        for i in range(6):
            instance = DemoClass.get()
            result = await instance.process(f"batch2_{i}")
            print(f"  Call {i + 1}: {result}")

        # Step 3: Scale down to 1 instance
        print("\n[Step 3] Scale down to 1 instance")
        print("  → Shutting down 2 server processes...")
        await DemoClass.deploy(instances=1, prefix="node")
        print("  ✓ 1 server process running (2 shut down)\n")

        # Test with 1 server
        print("Testing with 1 server:")
        for i in range(3):
            instance = DemoClass.get()
            result = await instance.process(f"batch3_{i}")
            print(f"  Call {i + 1}: {result}")

        # Step 4: Scale up again to 4 instances
        print("\n[Step 4] Scale up to 4 instances")
        print("  → Spawning 3 additional server processes...")
        await DemoClass.deploy(instances=4, prefix="node")
        print("  ✓ 4 server processes running (3 new, 1 existing)\n")

        # Test with 4 servers
        print("Testing round-robin across 4 servers:")
        for i in range(8):
            instance = DemoClass.get()
            result = await instance.process(f"batch4_{i}")
            print(f"  Call {i + 1}: {result}")

        # Step 5: No-op (already at 4)
        print("\n[Step 5] Deploy with same count (no-op)")
        print("  → Already at 4 instances, no changes...")
        await DemoClass.deploy(instances=4, prefix="node")
        print("  ✓ Still 4 server processes running\n")

        print("=" * 60)
        print("Summary:")
        print("  - Initial: 2 servers spawned")
        print("  - Scale up: +1 server (total 3)")
        print("  - Scale down: -2 servers (total 1)")
        print("  - Scale up: +3 servers (total 4)")
        print("  - No-op: 0 servers (total 4)")
        print("=" * 60)

        await DemoClass.shutdown()
        print("\nAll server processes shut down. Demo complete!")

    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("Please install: pip install 'thinkagain[grpc]'")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

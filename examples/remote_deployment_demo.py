"""Demo of remote multi-node deployment via SSH.

This example demonstrates deploying replicas across multiple machines using SSH:

1. Configure remote nodes with SSH credentials
2. Deploy replicas across localhost + remote workers
3. Automatic SSH connection and file transfer
4. Remote process spawning with environment setup

Prerequisites:
- SSH access to remote machines (key-based authentication)
- Python 3.9+ installed on all nodes
- Same Python environment on all nodes (or use env_setup to activate venv)
- Network connectivity between nodes for gRPC communication
- Install thinkagain with grpc extras: pip install 'thinkagain[grpc]'

Example remote setup:
    # On remote machine worker1.example.com
    $ python3 -m venv /opt/thinkagain-env
    $ source /opt/thinkagain-env/bin/activate
    $ pip install 'thinkagain[grpc]'

    # Ensure SSH key-based auth is set up
    $ ssh-copy-id -i ~/.ssh/id_rsa.pub user@worker1.example.com
"""

import asyncio

from thinkagain.distributed import init, replica
from thinkagain.distributed.nodes import NodeConfig


@replica(cpus=2, gpus=0)
class DataProcessor:
    """A CPU-intensive data processing service."""

    def __init__(self, node_name: str = "unknown"):
        self.node_name = node_name
        self.processed_count = 0

    def process(self, data: str) -> dict:
        """Process data and return result with metadata."""
        self.processed_count += 1
        return {
            "result": f"processed_{data}",
            "node": self.node_name,
            "count": self.processed_count,
        }


@replica(cpus=1, gpus=1)
class GPUModel:
    """A GPU-intensive model inference service."""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.inference_count = 0

    def infer(self, input_data: str) -> dict:
        """Run model inference."""
        self.inference_count += 1
        return {
            "prediction": f"{self.model_name}_result_{input_data}",
            "confidence": 0.95,
            "count": self.inference_count,
        }


async def main():
    print("=" * 70)
    print("Remote Multi-Node Deployment Demo")
    print("=" * 70)

    # ========================================================================
    # CONFIGURATION: Update these with your actual remote nodes
    # ========================================================================

    # Example 1: Single local node (no SSH required)
    local_only_nodes = [
        NodeConfig(
            host="localhost",
            cpus=8,
            gpus=0,
        )
    ]

    # Example 2: Localhost + remote workers (SSH required)
    # UNCOMMENT and UPDATE with your actual remote hosts:
    """
    multi_node_cluster = [
        # Local node
        NodeConfig(
            host="localhost",
            cpus=8,
            gpus=0,
        ),
        # Remote CPU worker
        NodeConfig(
            host="worker1.example.com",  # Update with your hostname
            cpus=16,
            gpus=0,
            ssh_user="ubuntu",  # Update with your SSH username
            ssh_key_path="~/.ssh/id_rsa",  # Path to your SSH private key
            ssh_port=22,
            python_executable="python3",
            # Optional: activate virtual environment on remote node
            env_setup="source /opt/thinkagain-env/bin/activate",
        ),
        # Remote GPU worker
        NodeConfig(
            host="gpu-worker.example.com",  # Update with your hostname
            cpus=32,
            gpus=4,
            ssh_user="ubuntu",
            ssh_key_path="~/.ssh/id_rsa",
            python_executable="python3",
            env_setup="source /opt/thinkagain-env/bin/activate",
        ),
    ]
    """

    # ========================================================================
    # Choose your configuration
    # ========================================================================

    # For testing locally:
    nodes = local_only_nodes

    # For actual remote deployment, uncomment:
    # nodes = multi_node_cluster

    try:
        # Initialize backend with node configuration
        print(f"\n[Init] Initializing cluster with {len(nodes)} node(s):")
        for node in nodes:
            node_type = "local" if node.is_local else "remote"
            print(f"  - {node.host}: {node.cpus} CPUs, {node.gpus} GPUs ({node_type})")

        init(backend="grpc", nodes=nodes)

        # ====================================================================
        # Deploy CPU service across available nodes
        # ====================================================================

        print("\n[Deploy] Deploying DataProcessor (cpus=2 per instance)")
        print("  → Scheduler will place instances on nodes with available CPUs...")

        await DataProcessor.deploy(instances=3, node_name="worker")
        print("  ✓ 3 DataProcessor instances deployed")

        # Get cluster state to see placement
        from thinkagain.distributed.runtime import get_backend

        backend = get_backend()
        cluster_state = backend.get_cluster_state()

        print("\n[Cluster State] Resource utilization after deployment:")
        for host, state in cluster_state.items():
            print(f"  {host}:")
            print(
                f"    CPUs: {state['available_cpus']}/{state['total_cpus']} available "
                f"({state['utilization']['cpu'] * 100:.1f}% used)"
            )
            print(
                f"    GPUs: {state['available_gpus']}/{state['total_gpus']} available "
                f"({state['utilization']['gpu'] * 100:.1f}% used)"
            )
            print(f"    Servers: {state['servers']}")

        # ====================================================================
        # Test the deployed service
        # ====================================================================

        print("\n[Test] Sending requests to DataProcessor instances:")
        for i in range(6):
            instance = DataProcessor.get()  # Round-robin
            result = await instance.process(f"item_{i}")
            print(f"  Request {i + 1}: {result}")

        # ====================================================================
        # Deploy GPU service (if GPU nodes are available)
        # ====================================================================

        has_gpu = any(node.gpus > 0 for node in nodes)
        if has_gpu:
            print("\n[Deploy] Deploying GPUModel (cpus=1, gpus=1 per instance)")
            print("  → Scheduler will prefer GPU-enabled nodes...")

            await GPUModel.deploy(instances=2, model_name="bert-base")
            print("  ✓ 2 GPUModel instances deployed")

            print("\n[Test] Sending requests to GPUModel instances:")
            for i in range(4):
                instance = GPUModel.get()
                result = await instance.infer(f"text_{i}")
                print(f"  Inference {i + 1}: {result}")

            # Show final cluster state
            cluster_state = backend.get_cluster_state()
            print("\n[Cluster State] Final resource utilization:")
            for host, state in cluster_state.items():
                print(f"  {host}:")
                print(
                    f"    CPUs: {state['available_cpus']}/{state['total_cpus']} "
                    f"({state['utilization']['cpu'] * 100:.1f}% used)"
                )
                print(
                    f"    GPUs: {state['available_gpus']}/{state['total_gpus']} "
                    f"({state['utilization']['gpu'] * 100:.1f}% used)"
                )
                print(f"    Servers: {state['servers']}")
        else:
            print("\n[Skip] No GPU nodes available, skipping GPUModel deployment")

        # ====================================================================
        # Cleanup
        # ====================================================================

        print("\n[Cleanup] Shutting down all services...")
        await DataProcessor.shutdown()
        if has_gpu:
            await GPUModel.shutdown()

        print("  ✓ All services shut down")
        print("\n" + "=" * 70)
        print("Demo complete!")
        print("=" * 70)

    except ImportError as e:
        print(f"\n❌ Error: Missing dependency - {e}")
        print("Please install: pip install 'thinkagain[grpc]'")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

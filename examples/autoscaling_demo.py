"""Demo of automatic replica scaling based on load.

This example demonstrates:
1. Initial deployment with minimal instances
2. Auto-scaler monitoring and adjusting instance counts
3. Dynamic scaling based on simulated load changes
4. Status monitoring and scaling event history

The auto-scaler runs as a background task, periodically checking profiling
data and adjusting replica counts to maintain target utilization.
"""

import asyncio
import time

from thinkagain import Context, node
from thinkagain.distributed import AutoScaler, ScalingPolicy, init, replica
from thinkagain.distributed.optimizer import batched_throughput, linear_throughput
from thinkagain.distributed.profiling import node_context, profile

# ============================================================================
# Define Replica Services
# ============================================================================


@replica(cpus=2)
class VectorDB:
    """Vector database - CPU-bound, linear scaling."""

    def __init__(self):
        self.search_count = 0

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Search for documents."""
        self.search_count += 1
        # Simulate work
        time.sleep(0.001)
        return [f"doc_{i}_{query}" for i in range(top_k)]


@replica(cpus=1, gpus=1)
class LLMPool:
    """LLM inference pool - GPU-bound, batching benefits."""

    def __init__(self):
        self.generation_count = 0

    def generate(self, prompt: str, context: list[str]) -> str:
        """Generate response from prompt and context."""
        self.generation_count += 1
        # Simulate work
        time.sleep(0.002)
        return f"Generated response for: {prompt[:20]}..."


@replica(cpus=1)
class Cache:
    """Simple cache - lightweight, high throughput."""

    def __init__(self):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> str | None:
        """Get value from cache."""
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        self.miss_count += 1
        return None

    def set(self, key: str, value: str) -> None:
        """Set value in cache."""
        self.cache[key] = value


# ============================================================================
# Define Pipeline Nodes
# ============================================================================


@node
async def retrieve(query: str) -> dict:
    """Retrieve documents from vector DB with caching."""
    cache = Cache.get()
    cache_key = f"docs:{query}"

    # Check cache
    cached = await cache.get(cache_key)
    if cached:
        return {"query": query, "docs": cached.split(",")}

    # Cache miss - fetch from VectorDB
    vdb = VectorDB.get()
    docs = await vdb.search(query, top_k=5)
    await cache.set(cache_key, ",".join(docs))

    return {"query": query, "docs": docs}


@node
async def generate(data: dict) -> str:
    """Generate response using LLM."""
    llm = LLMPool.get()
    response = await llm.generate(data["query"], data["docs"])
    return response


# ============================================================================
# Workload Simulator
# ============================================================================


async def simulate_workload(
    queries_per_second: float,
    duration: float,
):
    """Simulate incoming requests at specified rate."""
    print(f"  Simulating {queries_per_second} req/s for {duration}s...")

    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does gradient descent work?",
        "What is backpropagation?",
        "Describe transformers",
    ]

    start_time = time.time()
    request_count = 0

    while time.time() - start_time < duration:
        # Send request
        query = queries[request_count % len(queries)]

        # Run pipeline
        ctx = Context(query, context_factory=node_context)
        ctx = await retrieve(ctx)
        ctx = await generate(ctx)

        request_count += 1

        # Sleep to maintain target rate
        expected_time = start_time + (request_count / queries_per_second)
        sleep_time = expected_time - time.time()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    actual_rps = request_count / (time.time() - start_time)
    print(f"  Completed {request_count} requests ({actual_rps:.1f} req/s)")


# ============================================================================
# Main Demo
# ============================================================================


async def main():
    print("=" * 80)
    print("Auto-Scaling Demo")
    print("=" * 80)

    try:
        # Check for dependencies
        try:
            from thinkagain.distributed.optimizer import GUROBI_AVAILABLE

            if not GUROBI_AVAILABLE:
                print("\n⚠ WARNING: gurobipy not installed")
                print("Auto-scaling requires Gurobi optimizer.")
                print("Install with: pip install gurobipy")
                print("\nRunning demo without auto-scaling...")
                await run_without_autoscaling()
                return
        except ImportError:
            print("\n⚠ ERROR: Missing grpc dependencies")
            print("Install with: pip install 'thinkagain[grpc]'")
            return

        # ====================================================================
        # 1. Initialize Runtime
        # ====================================================================

        print("\n[1] Initializing distributed runtime...")
        init(backend="grpc")

        # ====================================================================
        # 2. Deploy Initial Instances (minimal)
        # ====================================================================

        print("\n[2] Deploying initial instances (minimal)...")
        await VectorDB.deploy(instances=1)
        await LLMPool.deploy(instances=1)
        await Cache.deploy(instances=1)
        print("  ✓ VectorDB: 1 instance")
        print("  ✓ LLMPool: 1 instance")
        print("  ✓ Cache: 1 instance")

        # ====================================================================
        # 3. Create Auto-Scaler
        # ====================================================================

        print("\n[3] Creating auto-scaler...")
        scaler = AutoScaler(
            target_rps=50.0,  # Start with moderate load
            check_interval=10.0,  # Check every 10 seconds
            cooldown_period=15.0,  # Wait 15s between scaling actions
            scale_up_threshold=0.75,  # Scale up if >75% utilized
            scale_down_threshold=0.40,  # Scale down if <40% utilized
            policies=[
                ScalingPolicy(
                    replica_name="VectorDB",
                    throughput_func=linear_throughput(200.0),  # 200 req/s per instance
                    cpus_per_instance=2,
                    min_instances=1,
                    max_instances=10,
                    max_utilization=0.8,
                ),
                ScalingPolicy(
                    replica_name="LLMPool",
                    throughput_func=batched_throughput(
                        base_rate=50.0,  # 50 req/s with batching
                        batch_size=32,
                    ),
                    cpus_per_instance=1,
                    gpus_per_instance=1,
                    min_instances=1,
                    max_instances=8,
                    max_utilization=0.8,
                ),
                ScalingPolicy(
                    replica_name="Cache",
                    throughput_func=linear_throughput(
                        1000.0
                    ),  # 1000 req/s per instance
                    cpus_per_instance=1,
                    min_instances=1,
                    max_instances=5,
                    max_utilization=0.8,
                ),
            ],
        )

        print("  ✓ Auto-scaler configured:")
        print(f"    - Target RPS: {scaler.target_rps}")
        print(f"    - Check interval: {scaler.check_interval}s")
        print(f"    - Policies: {len(scaler.policies)}")

        # ====================================================================
        # 4. Start Auto-Scaler
        # ====================================================================

        print("\n[4] Starting auto-scaler (background task)...")
        await scaler.start()
        print("  ✓ Auto-scaler running")

        # ====================================================================
        # 5. Run Workload with Profiling
        # ====================================================================

        print("\n[5] Running workload scenarios...")
        print("-" * 80)

        with profile():
            # Scenario 1: Moderate load
            print("\n[Scenario 1] Moderate load (30 req/s for 30s)")
            await simulate_workload(30.0, 30.0)

            # Show status after scenario 1
            await asyncio.sleep(2)  # Let scaler react
            print("\n  Auto-scaler status:")
            status = scaler.get_status()
            for name, s in status.items():
                print(
                    f"    {name:12}: {s.current_instances} instances "
                    f"(target: {s.target_instances}, util: {s.utilization:.1%})"
                )

            # Scenario 2: High load
            print("\n[Scenario 2] High load (100 req/s for 30s)")
            scaler.target_rps = 100.0  # Increase target
            await simulate_workload(100.0, 30.0)

            # Show status after scenario 2
            await asyncio.sleep(2)
            print("\n  Auto-scaler status:")
            status = scaler.get_status()
            for name, s in status.items():
                print(
                    f"    {name:12}: {s.current_instances} instances "
                    f"(target: {s.target_instances}, util: {s.utilization:.1%})"
                )

            # Scenario 3: Return to moderate
            print("\n[Scenario 3] Back to moderate load (30 req/s for 20s)")
            scaler.target_rps = 30.0  # Decrease target
            await simulate_workload(30.0, 20.0)

            # Final status
            await asyncio.sleep(2)
            print("\n  Auto-scaler final status:")
            status = scaler.get_status()
            for name, s in status.items():
                print(
                    f"    {name:12}: {s.current_instances} instances "
                    f"(target: {s.target_instances}, util: {s.utilization:.1%})"
                )

        # ====================================================================
        # 6. Show Scaling Report
        # ====================================================================

        print("\n[6] Auto-scaling report")
        print("-" * 80)

        report = scaler.get_report()
        print(f"  Total instances: {report['total_instances']}")
        print(f"  Avg utilization: {report['avg_utilization']:.1%}")
        print(f"  Total scaling events: {report['total_scaling_events']}")

        if report["recent_events"]:
            print("\n  Recent scaling events:")
            for event in report["recent_events"]:
                print(
                    f"    [{event['time']}] {event['replica']:12}: "
                    f"{event['from']} → {event['to']} instances ({event['reason']})"
                )
        else:
            print("\n  No scaling events occurred")

        # ====================================================================
        # 7. Cleanup
        # ====================================================================

        print("\n[7] Cleaning up...")
        await scaler.stop()
        await VectorDB.shutdown()
        await LLMPool.shutdown()
        await Cache.shutdown()
        print("  ✓ All services shut down")

        print("\n" + "=" * 80)
        print("Demo Complete!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def run_without_autoscaling():
    """Fallback demo without auto-scaling."""
    print("\n[Running manual scaling demo instead...]\n")

    init(backend="grpc")

    await VectorDB.deploy(instances=2)
    await Cache.deploy(instances=1)

    print("Deployed services:")
    print("  - VectorDB: 2 instances")
    print("  - Cache: 1 instance")
    print("  (Skipping LLMPool - requires GPU)")

    with profile():
        print("\nRunning sample workload...")
        for i in range(20):
            ctx = Context(f"Query {i}", context_factory=node_context)
            ctx = await retrieve(ctx)
            # Skip generate since LLMPool is not deployed

        print("✓ Workload complete")

    await VectorDB.shutdown()
    await Cache.shutdown()
    print("✓ Services shut down")


if __name__ == "__main__":
    asyncio.run(main())

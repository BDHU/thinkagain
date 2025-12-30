"""Combined profiling + optimization demo for replica-based services.

This example demonstrates:
1. How to profile replica calls during @node execution
2. How to inspect dependency graphs and fanout ratios
3. How to run the optimizer using the profiling data
"""

from thinkagain import node, run
from thinkagain.distributed import replica
from thinkagain.distributed.profiling import profiling_enabled

try:
    from thinkagain.distributed.optimizer import (
        Constraints,
        ReplicaConfig,
        batched_throughput,
        compute_optimal_counts_from_profiler,
        linear_throughput,
    )

    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("Warning: gurobipy not installed. Install with: pip install gurobipy")


# Define replica services with different characteristics
@replica(n=2)
class VectorDB:
    """Vector database with linear scaling."""

    def search(self, query: str, top_k: int = 3) -> list[str]:
        return [f"doc_{i}" for i in range(top_k)]


@replica(n=4)
class LLMPool:
    """LLM pool with batching effects."""

    def generate(self, prompt: str, context: list[str]) -> str:
        return f"Response based on: {', '.join(context[:2])}"


@replica(n=1)
class Cache:
    """Cache with linear scaling."""

    def __init__(self):
        self.store = {}

    def lookup(self, key: str) -> list[str] | None:
        return self.store.get(key)

    def set(self, key: str, value: list[str]) -> None:
        self.store[key] = value


# Define pipeline nodes
@node
async def retrieve_context(query: str) -> dict:
    """Retrieve from vector DB with a simple cache."""
    cache = Cache.get()
    vdb = VectorDB.get()
    cache_key = f"ctx_{query}"
    cached = cache.lookup(cache_key)
    if cached is None:
        cached = vdb.search(query)
        cache.set(cache_key, cached)

    return {"query": query, "docs": cached}


@node
async def generate_response(data: dict) -> str:
    """Generate response using LLM."""
    llm = LLMPool.get()
    response = llm.generate(data["query"], data["docs"])
    return response


def main():
    """Run the combined demo."""
    print("=" * 72)
    print("Replica Profiling + Optimization Demo")
    print("=" * 72)

    # Deploy replicas
    print("\n1. Deploying replicas...")
    VectorDB.deploy()
    LLMPool.deploy()
    Cache.deploy()

    # Define pipeline
    def pipeline(ctx):
        ctx = retrieve_context(ctx)
        ctx = generate_response(ctx)
        return ctx

    # Profile the workload
    print("\n2. Profiling workload...")
    print("-" * 72)

    with profiling_enabled() as session:
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?",
            "How does backpropagation work?",
        ]

        for query in queries:
            result = run(
                pipeline,
                query,
                context_factory=session.context_factory,
            )
            print(f"  {query[:32]:32} → {result.data[:40]}...")

        print("\n  Dependency graph:")
        deps = session.profiler.get_dependency_graph()
        for node_name, replicas in sorted(deps.items()):
            print(f"    {node_name:18} → {', '.join(sorted(replicas))}")

        print("\n  Fanout matrix:")
        fanout = session.profiler.get_fanout_matrix()
        for node_name, replica_fanout in sorted(fanout.items()):
            print(f"    {node_name}:")
            for replica, count in sorted(replica_fanout.items()):
                print(f"      {replica:12} : {count:.2f} calls/execution")

        summary = session.profiler.summary()
        print("\n  Call counts:")
        for replica, count in sorted(summary["call_counts"].items()):
            print(f"    {replica:12} : {count} total calls")

        print("\n  Node executions:")
        for node_name, count in sorted(summary["node_executions"].items()):
            print(f"    {node_name:18} : {count} executions")

        print(f"\n  Elapsed: {summary['elapsed_seconds']:.3f}s")

        # Optimizer section (optional)
        if OPTIMIZER_AVAILABLE:
            print("\n3. Optimizing replica counts...")
            print("-" * 72)

            configs = [
                ReplicaConfig(
                    name="VectorDB",
                    throughput_func=linear_throughput(200.0),
                    cost_per_instance=1.0,
                    base_service_rate=200.0,
                ),
                ReplicaConfig(
                    name="LLMPool",
                    throughput_func=batched_throughput(
                        base_rate=50.0,
                        batch_size=32,
                    ),
                    cost_per_instance=10.0,
                    base_service_rate=50.0,
                ),
                ReplicaConfig(
                    name="Cache",
                    throughput_func=linear_throughput(1000.0),
                    cost_per_instance=0.1,
                    base_service_rate=1000.0,
                ),
            ]

            scenarios = [
                ("Moderate traffic", 50.0),
                ("Higher traffic", 200.0),
            ]

            for name, external_rps in scenarios:
                print(f"\n  Scenario: {name} ({external_rps:.0f} req/s)")
                result = compute_optimal_counts_from_profiler(
                    session.profiler,
                    external_rps=external_rps,
                    replica_configs=configs,
                    constraints=Constraints(max_utilization=0.8),
                )

                if result.solver_status != "OPTIMAL":
                    print(f"    Status: {result.solver_status}")
                    continue

                print(f"    Total Cost: {result.total_cost:.2f}")
                print(f"    Total Instances: {result.total_instances}")
                for replica in sorted(result.replica_counts.keys()):
                    count = result.replica_counts[replica]
                    util = result.utilizations[replica]
                    capacity = result.capacities[replica]
                    arrival = result.arrival_rates[replica]
                    print(
                        f"    {replica:12} : {count:2} instances "
                        f"(util: {util:5.1%}, capacity: {capacity:6.1f}, "
                        f"arrival: {arrival:6.1f})"
                    )
        else:
            print("\n3. Optimizer skipped (gurobipy not installed).")

    # Cleanup
    print("\n4. Cleaning up...")
    VectorDB.shutdown()
    LLMPool.shutdown()
    Cache.shutdown()

    print("\n" + "=" * 72)
    print("Demo Complete!")
    print("=" * 72)


if __name__ == "__main__":
    main()

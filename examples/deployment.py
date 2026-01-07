"""Comprehensive deployment, scaling, and optimization demo.

Demonstrates:
- Basic manual scaling (scale up/down instances)
- Remote multi-node deployment via SSH
- Auto-scaling based on load with profiling
- Optimization with constraint solving

Run different scenarios:
    python deployment.py                    # Basic scaling
    python deployment.py --remote           # Remote deployment
    python deployment.py --autoscale        # Auto-scaling with load
    python deployment.py --optimize         # Profiling + optimizer

Requirements:
    pip install 'thinkagain[grpc]'          # For all modes
    pip install gurobipy                    # For --autoscale and --optimize
"""

import argparse
import asyncio
import time

from thinkagain import Context, node, run
from thinkagain.distributed import init, replica
from thinkagain.distributed.profiling import node_context, profile


# ============================================================================
# Services
# ============================================================================


@replica(cpus=2)
class VectorDB:
    def __init__(self):
        self.count = 0

    def search(self, query: str, top_k: int = 3) -> list[str]:
        self.count += 1
        time.sleep(0.001)
        return [f"doc_{i}" for i in range(top_k)]


@replica(cpus=1, gpus=1)
class LLM:
    def __init__(self):
        self.count = 0

    def generate(self, prompt: str, docs: list[str]) -> str:
        self.count += 1
        time.sleep(0.002)
        return f"Response for: {prompt[:30]}..."


@replica(cpus=1)
class Cache:
    def __init__(self):
        self.store = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str):
        self.store[key] = value


# ============================================================================
# Pipeline
# ============================================================================


@node
async def retrieve(query: str) -> dict:
    cache = Cache.get()
    key = f"docs:{query}"
    cached = await cache.get(key)

    if cached:
        return {"query": query, "docs": cached.split(",")}

    vdb = VectorDB.get()
    docs = await vdb.search(query)
    await cache.set(key, ",".join(docs))
    return {"query": query, "docs": docs}


@node
async def generate_response(data: dict) -> str:
    llm = LLM.get()
    return await llm.generate(data["query"], data["docs"])


def pipeline(ctx):
    ctx = retrieve(ctx)
    ctx = generate_response(ctx)
    return ctx


# ============================================================================
# Basic Scaling
# ============================================================================


async def demo_basic():
    """Simple manual scaling demo."""
    print("=" * 70)
    print("BASIC SCALING")
    print("=" * 70)

    init(backend="grpc")

    print("\n[1] Deploy 2 instances")
    await VectorDB.deploy(instances=2)

    print("Testing 2 instances:")
    for i in range(4):
        result = await VectorDB.get().search(f"q{i}")
        print(f"  {i + 1}: {len(result)} docs")

    print("\n[2] Scale up to 4 instances")
    await VectorDB.deploy(instances=4)

    print("Testing 4 instances:")
    for i in range(4):
        result = await VectorDB.get().search(f"q{i}")
        print(f"  {i + 1}: {len(result)} docs")

    print("\n[3] Scale down to 1 instance")
    await VectorDB.deploy(instances=1)

    print("Testing 1 instance:")
    for i in range(2):
        result = await VectorDB.get().search(f"q{i}")
        print(f"  {i + 1}: {len(result)} docs")

    await VectorDB.shutdown()
    print("\n✓ Done\n" + "=" * 70)


# ============================================================================
# Remote Deployment
# ============================================================================


async def demo_remote():
    """Multi-node remote deployment demo."""
    print("=" * 70)
    print("REMOTE DEPLOYMENT")
    print("=" * 70)

    from thinkagain.distributed.nodes import NodeConfig

    # Configure nodes (update with your actual hosts)
    nodes = [
        NodeConfig(host="localhost", cpus=8, gpus=0),
        # NodeConfig(
        #     host="worker1.example.com",
        #     cpus=16, gpus=2,
        #     ssh_user="ubuntu",
        #     ssh_key_path="~/.ssh/id_rsa",
        # ),
    ]

    print(f"\n[Cluster] {len(nodes)} node(s):")
    for n in nodes:
        print(f"  {n.host}: {n.cpus} CPUs, {n.gpus} GPUs")

    init(backend="grpc", nodes=nodes)

    print("\n[Deploy] Across cluster...")
    await VectorDB.deploy(instances=3)
    await Cache.deploy(instances=1)

    # Show placement
    from thinkagain.distributed.runtime import get_backend
    state = get_backend().get_cluster_state()

    print("\n[Resources]")
    for host, s in state.items():
        cpu_pct = s['utilization']['cpu'] * 100
        print(f"  {host}: {s['available_cpus']}/{s['total_cpus']} CPUs ({cpu_pct:.0f}% used)")

    print("\n[Test]")
    for i in range(4):
        result = await VectorDB.get().search(f"item_{i}")
        print(f"  {i + 1}: {len(result)} docs")

    await VectorDB.shutdown()
    await Cache.shutdown()
    print("\n✓ Done\n" + "=" * 70)


# ============================================================================
# Auto-Scaling
# ============================================================================


async def simulate_load(rps: float, duration: float):
    """Simulate workload at specified rate."""
    queries = ["ML basics", "Neural networks", "Gradient descent"]
    start = time.time()
    count = 0

    while time.time() - start < duration:
        q = queries[count % len(queries)]
        ctx = Context(q, context_factory=node_context)
        await retrieve(ctx)
        await generate_response(ctx)
        count += 1

        # Rate limiting
        expected = start + count / rps
        sleep = expected - time.time()
        if sleep > 0:
            await asyncio.sleep(sleep)

    print(f"  {count} requests in {time.time() - start:.1f}s")


async def demo_autoscale():
    """Auto-scaling based on load."""
    print("=" * 70)
    print("AUTO-SCALING")
    print("=" * 70)

    try:
        from thinkagain.distributed import AutoScaler, ScalingPolicy
        from thinkagain.distributed.optimizer import (
            GUROBI_AVAILABLE,
            batched_throughput,
            linear_throughput,
        )

        if not GUROBI_AVAILABLE:
            print("\n⚠ Requires: pip install gurobipy")
            return
    except ImportError as e:
        print(f"\n⚠ Error: {e}")
        return

    init(backend="grpc")

    print("\n[1] Initial deployment")
    await VectorDB.deploy(instances=1)
    await LLM.deploy(instances=1)
    await Cache.deploy(instances=1)

    print("\n[2] Configure auto-scaler")
    scaler = AutoScaler(
        target_rps=50.0,
        check_interval=8.0,
        cooldown_period=12.0,
        scale_up_threshold=0.75,
        scale_down_threshold=0.40,
        policies=[
            ScalingPolicy(
                replica_name="VectorDB",
                throughput_func=linear_throughput(200.0),
                cpus_per_instance=2,
                min_instances=1,
                max_instances=6,
                max_utilization=0.8,
            ),
            ScalingPolicy(
                replica_name="LLM",
                throughput_func=batched_throughput(50.0, 32),
                cpus_per_instance=1,
                gpus_per_instance=1,
                min_instances=1,
                max_instances=4,
                max_utilization=0.8,
            ),
            ScalingPolicy(
                replica_name="Cache",
                throughput_func=linear_throughput(1000.0),
                cpus_per_instance=1,
                min_instances=1,
                max_instances=3,
                max_utilization=0.8,
            ),
        ],
    )

    await scaler.start()

    print("\n[3] Run workload scenarios")
    with profile():
        print("\n  Low load (30 req/s, 15s)")
        await simulate_load(30.0, 15.0)
        await asyncio.sleep(1)

        status = scaler.get_status()
        for name, s in status.items():
            print(f"    {name}: {s.current_instances} inst, {s.utilization:.0%} util")

        print("\n  High load (100 req/s, 15s)")
        scaler.target_rps = 100.0
        await simulate_load(100.0, 15.0)
        await asyncio.sleep(1)

        status = scaler.get_status()
        for name, s in status.items():
            print(f"    {name}: {s.current_instances} inst, {s.utilization:.0%} util")

    report = scaler.get_report()
    print(f"\n[Report] {report['total_scaling_events']} scaling events")

    await scaler.stop()
    await VectorDB.shutdown()
    await LLM.shutdown()
    await Cache.shutdown()
    print("\n✓ Done\n" + "=" * 70)


# ============================================================================
# Optimizer
# ============================================================================


def demo_optimize():
    """Profiling and optimizer."""
    print("=" * 70)
    print("PROFILING + OPTIMIZER")
    print("=" * 70)

    try:
        from thinkagain.distributed.optimizer import (
            GUROBI_AVAILABLE,
            Constraints,
            ReplicaConfig,
            batched_throughput,
            compute_optimal_counts_from_profiler,
            linear_throughput,
        )

        if not GUROBI_AVAILABLE:
            print("\n⚠ Requires: pip install gurobipy")
            return
    except ImportError as e:
        print(f"\n⚠ Error: {e}")
        return

    print("\n[1] Deploy and profile")
    VectorDB.deploy(instances=2)
    LLM.deploy(instances=2)
    Cache.deploy(instances=1)

    with profile() as session:
        queries = [
            "What is ML?",
            "Explain neural networks",
            "How does backprop work?",
        ]

        for q in queries:
            result = run(pipeline, q, context_factory=session.context_factory)
            print(f"  {q[:30]:30} → {result.data[:25]}...")

        print("\n  Fanout:")
        for node_name, replicas in sorted(session.profiler.get_fanout_matrix().items()):
            for replica, count in sorted(replicas.items()):
                print(f"    {node_name} → {replica}: {count:.1f} calls/exec")

        print("\n[2] Optimize allocation")
        configs = [
            ReplicaConfig(
                name="VectorDB",
                throughput_func=linear_throughput(200.0),
                cpus_per_instance=2,
                base_service_rate=200.0,
            ),
            ReplicaConfig(
                name="LLM",
                throughput_func=batched_throughput(50.0, 32),
                cpus_per_instance=0,
                gpus_per_instance=1,
                base_service_rate=50.0,
            ),
            ReplicaConfig(
                name="Cache",
                throughput_func=linear_throughput(1000.0),
                cpus_per_instance=1,
                base_service_rate=1000.0,
            ),
        ]

        scenarios = [
            ("Moderate", 50.0, 64, 4),
            ("High", 200.0, 128, 8),
            ("CPU-limited", 100.0, 32, 8),
        ]

        for name, rps, cpus, gpus in scenarios:
            result = compute_optimal_counts_from_profiler(
                session.profiler,
                external_rps=rps,
                replica_configs=configs,
                constraints=Constraints(
                    max_utilization=0.8,
                    max_total_cpus=cpus,
                    max_total_gpus=gpus,
                ),
            )

            if result.solver_status == "OPTIMAL":
                print(f"\n  {name} ({rps} req/s): {result.total_cpus}/{cpus} CPUs, {result.total_gpus}/{gpus} GPUs")
                for r, count in sorted(result.replica_counts.items()):
                    util = result.utilizations[r]
                    print(f"    {r}: {count} inst ({util:.0%} util)")

    VectorDB.shutdown()
    LLM.shutdown()
    Cache.shutdown()
    print("\n✓ Done\n" + "=" * 70)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="Remote deployment")
    parser.add_argument("--autoscale", action="store_true", help="Auto-scaling")
    parser.add_argument("--optimize", action="store_true", help="Optimizer")
    args = parser.parse_args()

    if args.remote:
        asyncio.run(demo_remote())
    elif args.autoscale:
        asyncio.run(demo_autoscale())
    elif args.optimize:
        demo_optimize()
    else:
        asyncio.run(demo_basic())


if __name__ == "__main__":
    main()

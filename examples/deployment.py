"""Comprehensive deployment, scaling, and optimization demo.

Demonstrates the new clean optimizer-first API:
- Profile → Optimize → Deploy workflow
- Automatic optimal instance count computation
- Scenario comparison for capacity planning
- Auto-scaling with re-optimization

Run different scenarios:
    python deployment.py                    # Optimize and deploy
    python deployment.py --scenarios        # Compare deployment scenarios
    python deployment.py --autoscale        # Optimize + auto-scale

Requirements:
    pip install 'thinkagain[grpc]'          # For all modes
    pip install gurobipy                    # For optimization
"""

import argparse
import asyncio
import time

from thinkagain import Context, node, run
from thinkagain.distributed import (
    AutoScaler,
    Constraints,
    compare_scenarios,
    init,
    optimize,
    profile,
    replica,
)
from thinkagain.distributed.optimizer import batched_throughput, linear_throughput
from thinkagain.distributed.profiling import node_context


# ============================================================================
# Services
# ============================================================================


@replica(cpus=2, throughput=linear_throughput(200.0))
class VectorDB:
    def __init__(self):
        self.count = 0

    def search(self, query: str, top_k: int = 3) -> list[str]:
        self.count += 1
        time.sleep(0.001)
        return [f"doc_{i}" for i in range(top_k)]


@replica(cpus=1, gpus=1, throughput=batched_throughput(50.0, 32))
class LLM:
    def __init__(self):
        self.count = 0

    def generate(self, prompt: str, docs: list[str]) -> str:
        self.count += 1
        time.sleep(0.002)
        return f"Response for: {prompt[:30]}..."


@replica(cpus=1, throughput=linear_throughput(1000.0))
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
# Optimize and Deploy (New Clean API)
# ============================================================================


async def demo_optimize_deploy():
    """New clean API: Profile → Optimize → Deploy."""
    print("=" * 70)
    print("OPTIMIZE AND DEPLOY")
    print("=" * 70)

    try:
        from thinkagain.distributed.optimizer import GUROBI_AVAILABLE

        if not GUROBI_AVAILABLE:
            print("\n⚠ Requires: pip install gurobipy")
            return
    except ImportError as e:
        print(f"\n⚠ Error: {e}")
        return

    init(backend="grpc")

    print("\n[1] Profile sample workload")
    queries = [
        "What is ML?",
        "Explain neural networks",
        "How does backprop work?",
        "What are transformers?",
        "Explain gradient descent",
    ]

    with profile():
        # Deploy minimal instances for profiling
        await VectorDB.deploy(instances=1)
        await LLM.deploy(instances=1)
        await Cache.deploy(instances=1)

        # Run sample queries to profile the workflow
        for q in queries:
            result = run(pipeline, q, context_factory=node_context)
            print(f"  ✓ {q[:40]}")

        # Shutdown profiling deployment
        await VectorDB.shutdown()
        await LLM.shutdown()
        await Cache.shutdown()

    print("\n[2] Optimize deployment plan")
    plan = optimize(
        target_rps=100.0,
        max_cpus=64,
        max_gpus=4,
        max_utilization=0.75,
    )

    print("\n" + str(plan))

    print("\n[3] Deploy with optimal configuration")
    await plan.deploy()

    print("\n[4] Run production workload")
    for i, q in enumerate(queries * 2):
        result = run(pipeline, q, context_factory=node_context)
        if i % 3 == 0:
            print(f"  Request {i + 1}: {result.data[:40]}...")

    await VectorDB.shutdown()
    await LLM.shutdown()
    await Cache.shutdown()
    print("\n✓ Done\n" + "=" * 70)


# ============================================================================
# Scenario Comparison
# ============================================================================


async def demo_scenarios():
    """Compare different deployment scenarios."""
    print("=" * 70)
    print("SCENARIO COMPARISON")
    print("=" * 70)

    try:
        from thinkagain.distributed.optimizer import GUROBI_AVAILABLE

        if not GUROBI_AVAILABLE:
            print("\n⚠ Requires: pip install gurobipy")
            return
    except ImportError as e:
        print(f"\n⚠ Error: {e}")
        return

    init(backend="grpc")

    print("\n[1] Profile workload")
    queries = [
        "What is ML?",
        "Explain neural networks",
        "How does backprop work?",
    ]

    with profile():
        await VectorDB.deploy(instances=1)
        await LLM.deploy(instances=1)
        await Cache.deploy(instances=1)

        for q in queries:
            run(pipeline, q, context_factory=node_context)

        await VectorDB.shutdown()
        await LLM.shutdown()
        await Cache.shutdown()

    print("\n[2] Compare scenarios")
    scenarios = {
        "Budget": Constraints(max_total_cpus=32, max_total_gpus=2),
        "Balanced": Constraints(
            max_total_cpus=64, max_total_gpus=4, max_utilization=0.75
        ),
        "Performance": Constraints(
            max_utilization=0.6, max_total_cpus=128, max_total_gpus=8
        ),
    }

    plans = compare_scenarios(target_rps=100.0, scenarios=scenarios)

    for name, plan in plans.items():
        print(f"\n{name}:")
        print(f"  Status: {plan.result.solver_status}")
        print(f"  Resources: {plan.total_cpus} CPUs, {plan.total_gpus} GPUs")
        print(f"  Instances: {plan.result.total_instances} total")
        print(
            f"  Avg Utilization: {plan.result.metrics.get('avg_utilization', 0.0):.1%}"
        )
        print("  Replica counts:")
        for replica_name, count in sorted(plan.replica_counts.items()):
            util = plan.result.utilizations.get(replica_name, 0.0)
            print(f"    {replica_name}: {count} ({util:.1%} util)")

    print("\n✓ Done\n" + "=" * 70)


# ============================================================================
# Auto-Scaling with Re-Optimization
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
    """Auto-scaling with periodic re-optimization."""
    print("=" * 70)
    print("AUTO-SCALING WITH RE-OPTIMIZATION")
    print("=" * 70)

    try:
        from thinkagain.distributed.optimizer import GUROBI_AVAILABLE

        if not GUROBI_AVAILABLE:
            print("\n⚠ Requires: pip install gurobipy")
            return
    except ImportError as e:
        print(f"\n⚠ Error: {e}")
        return

    init(backend="grpc")

    print("\n[1] Profile and optimize initial deployment")
    with profile():
        await VectorDB.deploy(instances=1)
        await LLM.deploy(instances=1)
        await Cache.deploy(instances=1)

        # Quick profiling run
        for _ in range(10):
            ctx = Context("test query", context_factory=node_context)
            await retrieve(ctx)
            await generate_response(ctx)

        await VectorDB.shutdown()
        await LLM.shutdown()
        await Cache.shutdown()

    plan = optimize(target_rps=50.0, max_cpus=64, max_gpus=4)
    print("\nInitial plan:")
    for name, count in plan.replica_counts.items():
        print(f"  {name}: {count} instances")

    print("\n[2] Deploy and start auto-scaler")
    await plan.deploy()

    scaler = AutoScaler(
        target_rps=50.0,
        check_interval=8.0,
        cooldown_period=12.0,
        constraints=Constraints(max_total_cpus=64, max_total_gpus=4),
    )

    await scaler.start()

    print("\n[3] Run varying workload")
    with profile():
        print("\n  Low load (30 req/s, 12s)")
        await simulate_load(30.0, 12.0)
        await asyncio.sleep(2)

        status = scaler.get_status()
        for name, s in status.items():
            print(f"    {name}: {s.current_instances} inst, {s.utilization:.0%} util")

        print("\n  High load (100 req/s, 12s)")
        scaler.target_rps = 100.0
        await simulate_load(100.0, 12.0)
        await asyncio.sleep(2)

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
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", action="store_true", help="Compare scenarios")
    parser.add_argument("--autoscale", action="store_true", help="Auto-scaling demo")
    args = parser.parse_args()

    if args.scenarios:
        asyncio.run(demo_scenarios())
    elif args.autoscale:
        asyncio.run(demo_autoscale())
    else:
        asyncio.run(demo_optimize_deploy())


if __name__ == "__main__":
    main()

"""Benchmark to measure executor overhead improvements.

This demonstrates the performance improvements from:
1. Zero-overhead profiling when disabled
2. Lazy hook evaluation (empty list fast path)
3. Removal of distributed imports from hot paths
"""

import asyncio
import time

import thinkagain as ta


# Simple computation function
@ta.node
async def add(a: int, b: int) -> int:
    return a + b


@ta.node
async def multiply(a: int, b: int) -> int:
    return a * b


@ta.jit
async def simple_pipeline(x: int) -> int:
    """Simple pipeline with multiple ops."""
    y = await add(x, 1)
    z = await multiply(y, 2)
    w = await add(z, 3)
    return await multiply(w, 4)


async def benchmark_overhead():
    """Benchmark executor overhead."""
    print("=" * 60)
    print("EXECUTOR OVERHEAD BENCHMARK")
    print("=" * 60)
    print()

    # Warmup
    for _ in range(100):
        await simple_pipeline(5)

    # Benchmark: profiling disabled (should be fastest)
    iterations = 10_000
    start = time.perf_counter()
    for i in range(iterations):
        _ = await simple_pipeline(i)
    duration_no_profiling = time.perf_counter() - start

    print(f"✓ {iterations:,} iterations with profiling DISABLED")
    print(f"  Total time: {duration_no_profiling:.4f}s")
    print(f"  Avg per iteration: {duration_no_profiling / iterations * 1000:.4f}ms")
    print()

    # Benchmark: profiling enabled
    profiler = ta.enable_profiling()
    start = time.perf_counter()
    for i in range(iterations):
        _ = await simple_pipeline(i)
    duration_with_profiling = time.perf_counter() - start
    ta.disable_profiling()

    print(f"✓ {iterations:,} iterations with profiling ENABLED")
    print(f"  Total time: {duration_with_profiling:.4f}s")
    print(f"  Avg per iteration: {duration_with_profiling / iterations * 1000:.4f}ms")
    print()

    # Calculate overhead
    overhead_pct = (
        (duration_with_profiling - duration_no_profiling) / duration_no_profiling * 100
    )

    print("SUMMARY")
    print("-" * 60)
    print(f"Profiling overhead: {overhead_pct:.1f}%")
    print(
        f"Speedup when disabled: {duration_with_profiling / duration_no_profiling:.2f}x"
    )
    print()

    # Show profiling stats
    profiler = ta.enable_profiling()
    for i in range(1000):
        await simple_pipeline(i)

    stats = profiler.get_execution_stats()
    print("Sample profiling stats (1000 runs):")
    print("-" * 60)
    for func_name, metrics in sorted(stats.items()):
        if "add" in func_name or "multiply" in func_name:
            print(f"  {func_name:30s} mean={metrics['mean'] * 1000:.3f}ms")

    ta.disable_profiling()


if __name__ == "__main__":
    asyncio.run(benchmark_overhead())

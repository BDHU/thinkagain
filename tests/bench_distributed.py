"""Benchmarks for distributed execution with @replica, @node, Mesh, and .go() API."""

import asyncio

import pytest

import thinkagain as ta


# =============================================================================
# Module-level replica classes for benchmarking
# =============================================================================


@ta.service()
class BasicProcessor:
    """Basic processor for general execution benchmarks."""

    def __init__(self, multiplier: int = 2):
        self.multiplier = multiplier
        self.calls = []

    async def process(self, x: int) -> int:
        """Process input by multiplying with multiplier."""
        self.calls.append(x)
        return x * self.multiplier


@ta.service()
class SlowProcessor:
    """Processor with async delay for profiling benchmarks."""

    async def process(self, x: int) -> int:
        await asyncio.sleep(0.0001)  # Small delay for realistic async
        return x * 2


@ta.service()
class Counter:
    """Stateful counter for state management benchmarks."""

    def __init__(self, start: int = 0):
        self.count = start

    async def increment(self) -> int:
        self.count += 1
        return self.count


# Node functions for composition
@ta.op
async def preprocess(x: int) -> int:
    """Simple preprocessing node."""
    return x + 1


@ta.op
async def postprocess(x: int) -> int:
    """Simple postprocessing node."""
    return x - 1


@ta.op
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Fixtures
@pytest.fixture(autouse=True)
def cleanup_pools():
    """Clean up pools before and after each test."""
    from thinkagain.runtime.pool import _get_pools

    pools = _get_pools()
    pools.clear()
    yield
    pools.clear()


@pytest.fixture(autouse=True)
def cleanup_profiling():
    """Disable profiling before and after each test."""
    ta.disable_profiling()
    yield
    ta.disable_profiling()


# =============================================================================
# Basic Replica Execution Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_basic_replica_execution(benchmark):
    """Benchmark basic replica execution with mesh using .go()."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def execute():
        with mesh:
            return await processor.process.go(5)

    result = await benchmark(execute)
    assert result == 10


@pytest.mark.asyncio
async def test_bench_replica_multiple_calls(benchmark):
    """Benchmark multiple calls to replica using .go()."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_multiple():
        results = []
        with mesh:
            for i in range(50):
                result = await processor.process.go(i)
                results.append(result)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 50


@pytest.mark.asyncio
async def test_bench_replica_with_different_multipliers(benchmark):
    """Benchmark chaining multiple replicas with .go()."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(5)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def execute():
        with mesh:
            result = await processor1.process.go(5)  # 5 * 2 = 10
            result = await processor2.process.go(result)  # 10 * 5 = 50
            return result

    result = await benchmark(execute)
    assert result == 50


# =============================================================================
# Mixed Pipeline Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_mixed_replicas_and_nodes(benchmark):
    """Benchmark pipeline with both replicas and regular nodes using .go()."""
    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_multiple():
        results = []
        with mesh:
            for i in range(20):
                x = await preprocess.go(i)
                x = await processor.process.go(x)
                x = await postprocess.go(x)
                results.append(x)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 20


@pytest.mark.asyncio
async def test_bench_complex_distributed_pipeline(benchmark):
    """Benchmark complex pipeline with multiple replicas and nodes using .go()."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(3)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_multiple():
        results = []
        with mesh:
            for i in range(30):
                x = await add.go(i, 1)
                result1 = await processor1.process.go(x)
                result2 = await processor2.process.go(x)
                sum_result = await add.go(result1, result2)
                final = await add.go(sum_result, 10)
                results.append(final)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 30


# =============================================================================
# Profiling Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_profiling_overhead(benchmark):
    """Benchmark profiling overhead during execution."""
    processor = SlowProcessor.init()  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_with_profiling():
        with ta.profile() as profiler:
            with mesh:
                for i in range(20):
                    await processor.process.go(i)
        return profiler.summary()

    summary = await benchmark(run_with_profiling)
    assert "execution_stats" in summary


@pytest.mark.asyncio
async def test_bench_profiling_summary_generation(benchmark):
    """Benchmark profiler summary generation."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    # Run pipeline first
    with ta.profile() as profiler:
        with mesh:
            for i in range(50):
                await processor.process.go(i)

    # Benchmark summary generation only
    summary = benchmark(profiler.summary)
    assert summary["elapsed_seconds"] > 0


# =============================================================================
# State Management Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_stateful_replica(benchmark):
    """Benchmark stateful replica execution using .go()."""
    counter = Counter.init(start=0)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(1)])

    async def run_multiple():
        results = []
        with mesh:
            for _ in range(100):
                result = await counter.increment.go()
                results.append(result)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 100


# =============================================================================
# Locally Defined Replica Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_locally_defined_replica(benchmark):
    """Benchmark locally-defined replica execution with .go()."""

    @ta.service()
    class LocalProcessor:
        def __init__(self, multiplier: int):
            self.multiplier = multiplier

        async def process(self, x: int) -> int:
            return x * self.multiplier

    processor = LocalProcessor.init(multiplier=3)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(1)])

    async def run_multiple():
        results = []
        with mesh:
            for i in range(30):
                result = await processor.process.go(i)
                results.append(result)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 30


# =============================================================================
# Multiple Replica Handles Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_multiple_replica_handles(benchmark):
    """Benchmark using multiple replica handles in a pipeline with .go()."""
    processors = [BasicProcessor.init(i + 2) for i in range(5)]  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def execute():
        with mesh:
            x = 1
            for proc in processors:
                x = await proc.process.go(x)
            return x

    result = await benchmark(execute)
    assert result > 0


@pytest.mark.asyncio
async def test_bench_replica_handle_switching(benchmark):
    """Benchmark switching between different replica handles with .go()."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(10)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_alternating():
        with mesh:
            for i in range(20):
                if i % 2 == 0:
                    await processor1.process.go(i)
                else:
                    await processor2.process.go(i)

    await benchmark(run_alternating)


# =============================================================================
# Mesh Configuration Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_mesh_context_overhead(benchmark):
    """Benchmark mesh context manager overhead with .go()."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_with_context_switching():
        # Benchmark with mesh context switching
        for i in range(10):
            with mesh:
                await processor.process.go(i)

    await benchmark(run_with_context_switching)


@pytest.mark.asyncio
async def test_bench_concurrent_replica_calls(benchmark):
    """Benchmark concurrent calls to the same replica using .go()."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_concurrent():
        with mesh:
            # Execute multiple calls concurrently
            tasks = [processor.process.go(i) for i in range(20)]
            results = await asyncio.gather(*tasks)
        return results

    results = await benchmark(run_concurrent)
    assert len(results) == 20


@pytest.mark.asyncio
async def test_bench_parallel_node_execution(benchmark):
    """Benchmark parallel node execution with .go()."""
    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_parallel():
        with mesh:
            # Submit multiple independent node calls
            tasks = [add.go(i, i + 1) for i in range(50)]
            results = await asyncio.gather(*tasks)
        return results

    results = await benchmark(run_parallel)
    assert len(results) == 50


@pytest.mark.asyncio
async def test_bench_mixed_parallel_sequential(benchmark):
    """Benchmark mixed parallel and sequential execution with .go()."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(3)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_mixed():
        with mesh:
            results = []
            for i in range(10):
                # Parallel execution
                ref1 = processor1.process.go(i)
                ref2 = processor2.process.go(i)
                r1, r2 = await asyncio.gather(ref1, ref2)

                # Sequential execution
                final = await add.go(r1, r2)
                results.append(final)
        return results

    results = await benchmark(run_mixed)
    assert len(results) == 10

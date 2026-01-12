"""Benchmarks for distributed execution with @replica, Mesh, and profiling."""

import asyncio

import pytest

import thinkagain as ta


# =============================================================================
# Module-level replica classes for benchmarking
# =============================================================================


@ta.replica()
class BasicProcessor:
    """Basic processor for general execution benchmarks."""

    def __init__(self, multiplier: int = 2):
        self.multiplier = multiplier
        self.calls = []

    async def __call__(self, x: int) -> int:
        """Process input by multiplying with multiplier."""
        self.calls.append(x)
        return x * self.multiplier


@ta.replica()
class SlowProcessor:
    """Processor with async delay for profiling benchmarks."""

    def __init__(self):
        pass

    async def __call__(self, x: int) -> int:
        await asyncio.sleep(0.0001)  # Small delay for realistic async
        return x * 2


@ta.replica()
class Counter:
    """Stateful counter for state management benchmarks."""

    def __init__(self, start: int = 0):
        self.count = start

    async def __call__(self) -> int:
        self.count += 1
        return self.count


# Fixtures
@pytest.fixture(autouse=True)
def cleanup_pools():
    """Clean up pools before and after each test."""
    from thinkagain.distributed.replication.pool import _get_pools

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
    """Benchmark basic replica execution with mesh."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    @ta.bind_service(processor=processor)
    @ta.node
    async def process(x: int) -> int:
        return await processor(x)

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await benchmark(pipeline, 5)

    assert result == 10


@pytest.mark.asyncio
async def test_bench_replica_multiple_calls(benchmark):
    """Benchmark multiple calls to replica."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    @ta.bind_service(processor=processor)
    @ta.node
    async def process(x: int) -> int:
        return await processor(x)

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_multiple():
        results = []
        with mesh:
            for i in range(50):
                result = await pipeline(i)
                results.append(result)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 50


@pytest.mark.asyncio
async def test_bench_replica_with_different_multipliers(benchmark):
    """Benchmark replica initialization with different parameters."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(5)  # type: ignore[attr-defined]

    @ta.bind_service(p1=processor1, p2=processor2)
    @ta.node
    async def process_both(x: int) -> int:
        result = await processor1(x)
        result = await processor2(result)
        return result

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process_both(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await benchmark(pipeline, 5)

    assert result == 50  # 5 * 2 * 5


# =============================================================================
# Mixed Pipeline Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_mixed_replicas_and_nodes(benchmark):
    """Benchmark pipeline with both replicas and regular nodes."""

    @ta.node
    async def preprocess(x: int) -> int:
        return x + 1

    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    @ta.bind_service(processor=processor)
    @ta.node
    async def process(x: int) -> int:
        return await processor(x)

    @ta.node
    async def postprocess(x: int) -> int:
        return x - 1

    @ta.jit
    async def pipeline(x: int) -> int:
        x = await preprocess(x)
        x = await process(x)
        x = await postprocess(x)
        return x

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_multiple():
        results = []
        with mesh:
            for i in range(20):
                result = await pipeline(i)
                results.append(result)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 20


@pytest.mark.asyncio
async def test_bench_complex_distributed_pipeline(benchmark):
    """Benchmark complex pipeline with multiple replicas and nodes."""

    @ta.node
    async def add(a: int, b: int) -> int:
        return a + b

    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(3)  # type: ignore[attr-defined]

    @ta.bind_service(p1=processor1, p2=processor2)
    @ta.node
    async def process_parallel(x: int) -> int:
        result1 = await processor1(x)
        result2 = await processor2(x)
        return await add(result1, result2)

    @ta.jit
    async def pipeline(x: int) -> int:
        x = await add(x, 1)
        x = await process_parallel(x)
        x = await add(x, 10)
        return x

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_multiple():
        results = []
        with mesh:
            for i in range(30):
                result = await pipeline(i)
                results.append(result)
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

    @ta.bind_service(processor=processor)
    @ta.node
    async def process(x: int) -> int:
        return await processor(x)

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_with_profiling():
        with ta.profile() as profiler:
            with mesh:
                for i in range(20):
                    await pipeline(i)
        return profiler.summary()

    summary = await benchmark(run_with_profiling)
    assert "execution_stats" in summary


@pytest.mark.asyncio
async def test_bench_profiling_summary_generation(benchmark):
    """Benchmark profiler summary generation."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    @ta.bind_service(processor=processor)
    @ta.node
    async def process(x: int) -> int:
        return await processor(x)

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    # Run pipeline first
    with ta.profile() as profiler:
        with mesh:
            for i in range(50):
                await pipeline(i)

    # Benchmark summary generation only
    summary = benchmark(profiler.summary)
    assert summary["elapsed_seconds"] > 0


# =============================================================================
# State Management Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_stateful_replica(benchmark):
    """Benchmark stateful replica execution."""
    counter = Counter.init(start=0)  # type: ignore[attr-defined]

    @ta.bind_service(counter=counter)
    @ta.node
    async def call_counter() -> int:
        return await counter()

    @ta.jit
    async def count_pipeline() -> int:
        return await call_counter()

    mesh = ta.Mesh([ta.CpuDevice(1)])

    async def run_multiple():
        results = []
        with mesh:
            for _ in range(100):
                result = await count_pipeline()
                results.append(result)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 100


@pytest.mark.asyncio
async def test_bench_apply_replica(benchmark):
    """Benchmark apply_replica for state updates."""

    class CounterClass:
        def __init__(self, count: int = 0, step: int = 1):
            self.count = count
            self.step = step

        def decompose(self):
            return [self.count, self.step], None

        @classmethod
        def compose(cls, aux, children):
            count, step = children
            return cls(count=count, step=step)

    @ta.node
    async def add(a: int, b: int) -> int:
        return a + b

    @ta.jit
    async def step(count: int, step: int, x: int):
        new_count = await add(count, step)
        out = await add(new_count, x)
        return [new_count, step], out

    async def run_apply_replica():
        counter = CounterClass()
        for i in range(50):
            await ta.apply_replica(counter, step, i)
        return counter.count

    count = await benchmark(run_apply_replica)
    assert count == 50


# =============================================================================
# Locally Defined Replica Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_locally_defined_replica(benchmark):
    """Benchmark locally-defined replica execution."""

    @ta.replica()
    class LocalProcessor:
        def __init__(self, multiplier: int):
            self.multiplier = multiplier

        async def __call__(self, x: int) -> int:
            return x * self.multiplier

    processor = LocalProcessor.init(multiplier=3)  # type: ignore[attr-defined]

    @ta.bind_service(processor=processor)
    @ta.node
    async def process(x: int) -> int:
        return await processor(x)

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(1)])

    async def run_multiple():
        results = []
        with mesh:
            for i in range(30):
                result = await pipeline(i)
                results.append(result)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 30


# =============================================================================
# Multiple Replica Handles Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_multiple_replica_handles(benchmark):
    """Benchmark using multiple replica handles in a pipeline."""
    processors = [BasicProcessor.init(i + 2) for i in range(5)]  # type: ignore[attr-defined]

    @ta.bind_service(
        p0=processors[0],
        p1=processors[1],
        p2=processors[2],
        p3=processors[3],
        p4=processors[4],
    )
    @ta.node
    async def process_sequential(x: int) -> int:
        x = await processors[0](x)
        x = await processors[1](x)
        x = await processors[2](x)
        x = await processors[3](x)
        x = await processors[4](x)
        return x

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process_sequential(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await benchmark(pipeline, 1)

    assert result > 0


@pytest.mark.asyncio
async def test_bench_replica_handle_switching(benchmark):
    """Benchmark switching between different replica handles."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(10)  # type: ignore[attr-defined]

    @ta.bind_service(p1=processor1)
    @ta.node
    async def process1(x: int) -> int:
        return await processor1(x)

    @ta.bind_service(p2=processor2)
    @ta.node
    async def process2(x: int) -> int:
        return await processor2(x)

    @ta.jit
    async def pipeline1(x: int) -> int:
        return await process1(x)

    @ta.jit
    async def pipeline2(x: int) -> int:
        return await process2(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_alternating():
        with mesh:
            for i in range(20):
                if i % 2 == 0:
                    await pipeline1(i)
                else:
                    await pipeline2(i)

    await benchmark(run_alternating)


# =============================================================================
# Mesh Configuration Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_mesh_context_overhead(benchmark):
    """Benchmark mesh context manager overhead."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    @ta.bind_service(processor=processor)
    @ta.node
    async def process(x: int) -> int:
        return await processor(x)

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_with_context_switching():
        # Benchmark with mesh context switching
        for i in range(10):
            with mesh:
                await pipeline(i)

    await benchmark(run_with_context_switching)


@pytest.mark.asyncio
async def test_bench_concurrent_replica_calls(benchmark):
    """Benchmark concurrent calls to the same replica."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]

    @ta.bind_service(processor=processor)
    @ta.node
    async def process(x: int) -> int:
        return await processor(x)

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    async def run_concurrent():
        with mesh:
            # Execute multiple calls concurrently
            tasks = [pipeline(i) for i in range(20)]
            results = await asyncio.gather(*tasks)
        return results

    results = await benchmark(run_concurrent)
    assert len(results) == 20

"""Benchmarks for distributed execution."""

import asyncio

from thinkagain import chain, node, replica, run


def test_bench_replica_deployment(benchmark):
    """Benchmark replica deployment overhead."""

    @replica(n=2)
    class Service:
        def __init__(self, value: int = 0):
            self.value = value

    async def deploy_and_shutdown_async():
        await Service.deploy(value=5)
        await Service.shutdown()

    benchmark(lambda: asyncio.run(deploy_and_shutdown_async()))


def test_bench_replica_get(benchmark):
    """Benchmark getting replica instances (round-robin)."""

    @replica(n=4)
    class FastService:
        def ping(self) -> bool:
            return True

    asyncio.run(FastService.deploy())

    def get_instance():
        return FastService.get()

    result = benchmark(get_instance)
    assert result.ping()
    asyncio.run(FastService.shutdown())


def test_bench_replica_method_call(benchmark):
    """Benchmark calling methods on replica instances."""

    @replica(n=2)
    class Calculator:
        def __init__(self, factor: int = 1):
            self.factor = factor

        def multiply(self, x: int) -> int:
            return x * self.factor

    asyncio.run(Calculator.deploy(factor=2))

    def call_method():
        return Calculator.get().multiply(100)

    result = benchmark(call_method)
    assert result == 200
    asyncio.run(Calculator.shutdown())


def test_bench_pipeline_with_replica(benchmark):
    """Benchmark pipeline execution with replica service."""

    @replica(n=2)
    class Processor:
        def __init__(self, delta: int = 5):
            self.delta = delta

        def process(self, x: int) -> int:
            return x + self.delta

    @node
    async def initial_value(_: int) -> int:
        return 10

    @node
    async def apply_processor(value: int) -> int:
        return Processor.get().process(value)

    @node
    async def double_it(value: int) -> int:
        return value * 2

    pipeline = chain(initial_value, apply_processor, double_it)

    asyncio.run(Processor.deploy(delta=5))

    def run_pipeline():
        return run(pipeline, 0)

    result = benchmark(run_pipeline)
    assert result.data == 30  # (10+5)*2
    asyncio.run(Processor.shutdown())


def test_bench_runtime_context_manager(benchmark):
    """Benchmark pipeline execution with runtime context (replicas pre-deployed)."""

    @replica(n=2)
    class Service:
        def ping(self) -> bool:
            return True

    @node
    async def call_service(_: int) -> bool:
        return Service.get().ping()

    pipeline = chain(call_service)

    # Deploy replicas once before benchmarking
    asyncio.run(Service.deploy())

    def run_pipeline():
        # Just benchmark pipeline execution, not deploy/shutdown overhead
        return run(pipeline, 0)

    result = benchmark(run_pipeline)
    assert result.data is True

    # Cleanup after all iterations
    asyncio.run(Service.shutdown())


def test_bench_round_robin_distribution(benchmark):
    """Benchmark round-robin distribution across multiple instances."""

    @replica(n=4)
    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self) -> int:
            self.count += 1
            return self.count

    asyncio.run(Counter.deploy())

    def distribute_calls():
        results = []
        for _ in range(10):
            results.append(Counter.get().increment())
        return results

    results = benchmark(distribute_calls)
    # Should have distributed across 4 instances
    assert len(results) == 10
    asyncio.run(Counter.shutdown())


def test_bench_local_init_overhead(benchmark):
    """Benchmark overhead of __local_init__ customization."""

    @replica(n=2)
    class CustomInit:
        @classmethod
        def __local_init__(cls, base_value: int):
            adjusted_value = base_value * 2
            return cls(adjusted_value)

        def __init__(self, value: int):
            self.value = value

        def get_value(self) -> int:
            return self.value

    async def deploy_custom_init_async():
        await CustomInit.deploy(base_value=10)
        await CustomInit.shutdown()

    benchmark(lambda: asyncio.run(deploy_custom_init_async()))


def test_bench_multiple_replica_services(benchmark):
    """Benchmark pipeline with multiple different replica services."""

    @replica(n=2)
    class Adder:
        def __init__(self, delta: int = 1):
            self.delta = delta

        def add(self, x: int) -> int:
            return x + self.delta

    @replica(n=2)
    class Multiplier:
        def __init__(self, factor: int = 2):
            self.factor = factor

        def multiply(self, x: int) -> int:
            return x * self.factor

    @node
    async def start_value(_: int) -> int:
        return 5

    @node
    async def add_step(value: int) -> int:
        return Adder.get().add(value)

    @node
    async def multiply_step(value: int) -> int:
        return Multiplier.get().multiply(value)

    pipeline = chain(start_value, add_step, multiply_step)

    async def deploy_services_async():
        await Adder.deploy(delta=3)
        await Multiplier.deploy(factor=4)

    asyncio.run(deploy_services_async())

    def run_pipeline():
        return run(pipeline, 0)

    result = benchmark(run_pipeline)
    assert result.data == 32  # (5+3)*4
    asyncio.run(Adder.shutdown())
    asyncio.run(Multiplier.shutdown())

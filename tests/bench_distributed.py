"""Benchmarks for distributed execution."""

import pytest

from thinkagain import node, replica, run
from thinkagain.distributed import runtime, clear_replica_registry, reset_backend


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure each test starts with an empty registry."""
    clear_replica_registry()
    reset_backend()
    yield
    clear_replica_registry()
    reset_backend()


def test_bench_replica_deployment(benchmark):
    """Benchmark replica deployment overhead."""

    @replica(n=2)
    class Service:
        def __init__(self, value: int = 0):
            self.value = value

    def deploy_and_shutdown():
        Service.deploy(value=5)
        Service.shutdown()

    benchmark(deploy_and_shutdown)


def test_bench_replica_get(benchmark):
    """Benchmark getting replica instances (round-robin)."""

    @replica(n=4)
    class FastService:
        def ping(self) -> bool:
            return True

    FastService.deploy()

    def get_instance():
        return FastService.get()

    result = benchmark(get_instance)
    assert result.ping()


def test_bench_replica_method_call(benchmark):
    """Benchmark calling methods on replica instances."""

    @replica(n=2)
    class Calculator:
        def __init__(self, factor: int = 1):
            self.factor = factor

        def multiply(self, x: int) -> int:
            return x * self.factor

    Calculator.deploy(factor=2)

    def call_method():
        return Calculator.get().multiply(100)

    result = benchmark(call_method)
    assert result == 200


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

    def pipeline(ctx):
        ctx = initial_value(ctx)
        ctx = apply_processor(ctx)
        ctx = double_it(ctx)
        return ctx

    Processor.deploy(delta=5)

    def run_pipeline():
        return run(pipeline, 0)

    result = benchmark(run_pipeline)
    assert result.data == 30  # (10+5)*2


def test_bench_runtime_context_manager(benchmark):
    """Benchmark runtime context manager overhead."""

    @replica(n=2)
    class Service:
        def ping(self) -> bool:
            return True

    @node
    async def call_service(_: int) -> bool:
        return Service.get().ping()

    def pipeline(ctx):
        ctx = call_service(ctx)
        return ctx

    def run_with_runtime():
        with runtime():
            return run(pipeline, 0)

    result = benchmark(run_with_runtime)
    assert result.data is True


def test_bench_round_robin_distribution(benchmark):
    """Benchmark round-robin distribution across multiple instances."""

    @replica(n=4)
    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self) -> int:
            self.count += 1
            return self.count

    Counter.deploy()

    def distribute_calls():
        results = []
        for _ in range(10):
            results.append(Counter.get().increment())
        return results

    results = benchmark(distribute_calls)
    # Should have distributed across 4 instances
    assert len(results) == 10


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

    def deploy_custom_init():
        CustomInit.deploy(base_value=10)
        CustomInit.shutdown()

    benchmark(deploy_custom_init)


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

    def pipeline(ctx):
        ctx = start_value(ctx)
        ctx = add_step(ctx)
        ctx = multiply_step(ctx)
        return ctx

    Adder.deploy(delta=3)
    Multiplier.deploy(factor=4)

    def run_pipeline():
        return run(pipeline, 0)

    result = benchmark(run_pipeline)
    assert result.data == 32  # (5+3)*4

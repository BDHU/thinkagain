"""Benchmarks for declarative API."""

from thinkagain import chain, run, node
from conftest import add_one, double, append_x


def test_bench_linear_pipeline(benchmark):
    """Benchmark a simple linear pipeline."""

    pipeline = chain(add_one, double)

    result = benchmark(lambda: run(pipeline, 5))
    assert result.data == 12


def test_bench_nested_pipeline(benchmark):
    """Benchmark nested function calls."""

    def add_twice(ctx):
        ctx = add_one(ctx)
        ctx = add_one(ctx)
        return ctx

    def pipeline(ctx):
        ctx = add_twice(ctx)
        ctx = double(ctx)
        return ctx

    result = benchmark(lambda: run(pipeline, 5))
    assert result.data == 14


def test_bench_conditional_pipeline(benchmark):
    """Benchmark pipeline with conditional branching."""

    @node
    async def add_ten(value: int) -> int:
        return value + 10

    def pipeline(ctx):
        ctx = add_one(ctx)
        if ctx.data > 5:
            ctx = add_ten(ctx)
        return ctx

    result = benchmark(lambda: run(pipeline, 10))
    assert result.data == 21


def test_bench_loop_pipeline(benchmark):
    """Benchmark pipeline with loops."""

    def pipeline(ctx):
        for _ in range(10):
            ctx = append_x(ctx)
        return ctx

    result = benchmark(lambda: run(pipeline, []))
    assert len(result.data) == 10


def test_bench_while_loop(benchmark):
    """Benchmark while loop with condition checking."""

    def pipeline(ctx):
        while ctx.data < 100:
            ctx = add_one(ctx)
        return ctx

    result = benchmark(lambda: run(pipeline, 0))
    assert result.data == 100


def test_bench_deep_nesting(benchmark):
    """Benchmark deeply nested pipeline operations."""

    def level3(ctx):
        ctx = add_one(ctx)
        ctx = add_one(ctx)
        return ctx

    def level2(ctx):
        ctx = level3(ctx)
        ctx = double(ctx)
        return ctx

    def level1(ctx):
        ctx = level2(ctx)
        ctx = add_one(ctx)
        return ctx

    result = benchmark(lambda: run(level1, 1))
    assert result.data == 7  # (1+1+1)*2+1

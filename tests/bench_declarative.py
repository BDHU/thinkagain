"""Benchmarks for declarative API."""

from thinkagain import run, node
from conftest import add_one, double, append_x


def test_bench_linear_pipeline(benchmark):
    """Benchmark a simple linear pipeline."""

    def pipeline(ctx):
        ctx = add_one(ctx)
        ctx = double(ctx)
        return ctx

    result = benchmark(lambda: run(pipeline, {"value": 5}))
    assert result.get("value") == 12


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

    result = benchmark(lambda: run(pipeline, {"value": 5}))
    assert result.get("value") == 14


def test_bench_conditional_pipeline(benchmark):
    """Benchmark pipeline with conditional branching."""

    @node
    async def log_high(ctx):
        ctx.set("logs", ["high"])
        return ctx

    def pipeline(ctx):
        ctx = add_one(ctx)
        if ctx.get("value") > 5:
            ctx = log_high(ctx)
        return ctx

    result = benchmark(lambda: run(pipeline, {"value": 10}))
    assert result.get("logs") == ["high"]


def test_bench_loop_pipeline(benchmark):
    """Benchmark pipeline with loops."""

    def pipeline(ctx):
        for _ in range(10):
            ctx = append_x(ctx)
        return ctx

    result = benchmark(lambda: run(pipeline, {}))
    assert len(result.get("logs")) == 10


def test_bench_while_loop(benchmark):
    """Benchmark while loop with condition checking."""

    def pipeline(ctx):
        while ctx.get("value") < 100:
            ctx = add_one(ctx)
        return ctx

    result = benchmark(lambda: run(pipeline, {"value": 0}))
    assert result.get("value") == 100


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

    result = benchmark(lambda: run(level1, {"value": 1}))
    assert result.get("value") == 7  # (1+1+1)*2+1

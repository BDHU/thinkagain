"""Benchmarks for declarative API."""

from thinkagain import run, node


# Test functions
@node
async def add_one(ctx):
    ctx.set("value", ctx.get("value", 0) + 1)
    return ctx


@node
async def double(ctx):
    ctx.set("value", ctx.get("value") * 2)
    return ctx


@node
async def append_x(ctx):
    ctx.set("logs", ctx.get("logs", []) + ["x"])
    return ctx


def test_bench_linear_pipeline(benchmark):
    """Benchmark a simple linear pipeline."""

    def pipeline(ctx):
        ctx = add_one(ctx)
        ctx = double(ctx)
        return ctx

    def run_pipeline():
        return run(pipeline, {"value": 5})

    result = benchmark(run_pipeline)
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

    def run_pipeline():
        return run(pipeline, {"value": 5})

    result = benchmark(run_pipeline)
    assert result.get("value") == 14


def test_bench_conditional_pipeline(benchmark):
    """Benchmark pipeline with conditional branching."""

    @node
    async def log_high(ctx):
        ctx.set("logs", ["high"])
        return ctx

    @node
    async def log_low(ctx):
        ctx.set("logs", ["low"])
        return ctx

    def pipeline(ctx):
        ctx = add_one(ctx)
        if ctx.get("value") > 5:
            ctx = log_high(ctx)
        else:
            ctx = log_low(ctx)
        return ctx

    def run_pipeline():
        return run(pipeline, {"value": 10})

    result = benchmark(run_pipeline)
    assert result.get("logs") == ["high"]


def test_bench_loop_pipeline(benchmark):
    """Benchmark pipeline with loops."""

    def pipeline(ctx):
        for _ in range(10):
            ctx = append_x(ctx)
        return ctx

    def run_pipeline():
        return run(pipeline, {})

    result = benchmark(run_pipeline)
    assert len(result.get("logs")) == 10


def test_bench_materialization(benchmark):
    """Benchmark auto-materialization behavior."""

    def pipeline(ctx):
        ctx = add_one(ctx)
        # Reading value forces materialization
        _ = ctx.get("value")
        ctx = double(ctx)
        return ctx

    def run_pipeline():
        return run(pipeline, {"value": 5})

    result = benchmark(run_pipeline)
    assert result.get("value") == 12


def test_bench_while_loop(benchmark):
    """Benchmark while loop with condition checking."""

    def pipeline(ctx):
        while ctx.get("value") < 100:
            ctx = add_one(ctx)
        return ctx

    def run_pipeline():
        return run(pipeline, {"value": 0})

    result = benchmark(run_pipeline)
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

    def pipeline(ctx):
        ctx = level1(ctx)
        return ctx

    def run_pipeline():
        return run(pipeline, {"value": 1})

    result = benchmark(run_pipeline)
    assert result.get("value") == 7  # (1+1+1)*2+1


def test_bench_metadata_tracking(benchmark):
    """Benchmark metadata collection overhead."""

    def pipeline(ctx):
        for _ in range(5):
            ctx = add_one(ctx)
            ctx = double(ctx)
        return ctx

    def run_pipeline():
        result = run(pipeline, {"value": 1})
        # Force metadata access
        _ = result.metadata.total_duration
        _ = result.metadata.node_latencies
        return result

    result = benchmark(run_pipeline)
    assert result.metadata.node_execution_count == 10

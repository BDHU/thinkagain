"""Benchmarks for JAX-style tracing and graph compilation."""

import pytest
from dataclasses import dataclass

import thinkagain
from thinkagain.core.tracing.tracer import _compiled_cache


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_compiled_cache():
    _compiled_cache.clear()


@dataclass
class State:
    value: int
    threshold: int = 0
    label: str = ""


@thinkagain.node
async def increment(state: State) -> State:
    """Increment state value by 1."""
    return State(
        value=state.value + 1, threshold=state.threshold, label=f"{state.label}_inc"
    )


@thinkagain.node
async def double(state: State) -> State:
    """Double state value."""
    return State(
        value=state.value * 2, threshold=state.threshold, label=f"{state.label}_double"
    )


@thinkagain.node
async def get_threshold(state: State) -> int:
    """Extract threshold from state."""
    return state.threshold


@thinkagain.node
async def increment_by_threshold(state: State, threshold: int) -> State:
    """Increment by captured threshold value."""
    return State(
        value=state.value + threshold, threshold=state.threshold, label=state.label
    )


@thinkagain.node
async def process_item(carry: int, item: str) -> tuple[int, str]:
    """Process item for scan tests."""
    new_carry = carry + 1
    result = f"Processed: {item} (#{carry + 1})"
    return new_carry, result


@thinkagain.node
async def add_one(x: int) -> int:
    """Add one to input."""
    return x + 1


@thinkagain.node
async def multiply(x: int, factor: int) -> int:
    """Multiply x by factor."""
    return x * factor


# =============================================================================
# Basic Execution Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_simple_pipeline(benchmark):
    """Benchmark simple linear pipeline execution."""

    @thinkagain.jit
    async def pipeline(x: int) -> int:
        x = await add_one(x)
        x = await add_one(x)
        x = await add_one(x)
        return x

    result = await benchmark(pipeline, 5)
    assert result == 8


@pytest.mark.asyncio
async def test_bench_pipeline_with_kwargs(benchmark):
    """Benchmark pipeline with keyword arguments."""

    @thinkagain.jit
    async def pipeline(x: int, factor: int) -> int:
        x = await add_one(x)
        x = await multiply(x, factor)
        return x

    result = await benchmark(pipeline, 5, 3)
    assert result == 18


@pytest.mark.asyncio
async def test_bench_graph_compilation(benchmark):
    """Benchmark graph compilation (first call only)."""

    @thinkagain.jit
    async def pipeline(x: int) -> int:
        x = await add_one(x)
        x = await multiply(x, 2)
        x = await add_one(x)
        return x

    # First call compiles the graph
    result = await benchmark(pipeline, 10)
    assert result == 23


@pytest.mark.asyncio
async def test_bench_cached_graph_execution(benchmark):
    """Benchmark execution with cached graph (no compilation overhead)."""

    @thinkagain.jit
    async def pipeline(x: int) -> int:
        x = await add_one(x)
        x = await multiply(x, 2)
        x = await add_one(x)
        return x

    # Warm up cache
    await pipeline(10)

    # Benchmark cached execution
    result = await benchmark(pipeline, 20)
    assert result == 43


# =============================================================================
# Control Flow Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_cond_execution(benchmark):
    """Benchmark conditional branching execution."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        state = await thinkagain.cond(  # type: ignore[assignment]
            lambda s: s.value < 10,
            increment,
            double,
            state,
        )
        return state

    result = await benchmark(pipeline, State(value=5))
    assert result.value == 6


@pytest.mark.asyncio
async def test_bench_while_loop(benchmark):
    """Benchmark while loop execution."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        state = await thinkagain.while_loop(  # type: ignore[assignment]
            lambda s: s.value < 100,
            increment,
            state,
        )
        return state

    result = await benchmark(pipeline, State(value=0))
    assert result.value == 100


@pytest.mark.asyncio
async def test_bench_scan(benchmark):
    """Benchmark scan operation."""

    @thinkagain.jit
    async def pipeline(items: list[str]) -> list[str]:
        final_count, results = await thinkagain.scan(
            process_item,
            init=0,
            xs=items,
        )
        return results  # type: ignore[return-value]

    items = [f"item_{i}" for i in range(20)]
    results = await benchmark(pipeline, items)
    assert len(results) == 20


@pytest.mark.asyncio
async def test_bench_switch(benchmark):
    """Benchmark switch operator."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        return await thinkagain.switch(  # type: ignore[return-value]
            lambda s: 0 if s.value < 10 else 1,
            [increment, double, increment, double],
            state,
        )

    result = await benchmark(pipeline, State(value=10))
    assert result.value == 20


# =============================================================================
# Captured Values Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_captured_value_in_cond(benchmark):
    """Benchmark cond with captured values."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        threshold = await get_threshold(state)

        @thinkagain.node
        async def high_branch(s: State) -> State:
            return await increment_by_threshold(s, threshold)

        state = await thinkagain.cond(  # type: ignore[assignment]
            lambda s: s.value < 10,
            high_branch,
            increment,
            state,
        )
        return state

    result = await benchmark(pipeline, State(value=5, threshold=3))
    assert result.value == 8


@pytest.mark.asyncio
async def test_bench_captured_value_in_while(benchmark):
    """Benchmark while loop with captured values."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        threshold = await get_threshold(state)

        @thinkagain.node
        async def body_with_capture(s: State) -> State:
            return await increment_by_threshold(s, threshold)

        state = await thinkagain.while_loop(  # type: ignore[assignment]
            lambda s: s.value < 15,
            body_with_capture,
            state,
        )
        return state

    result = await benchmark(pipeline, State(value=0, threshold=5))
    assert result.value == 15


# =============================================================================
# Complex Pipeline Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_nested_control_flow(benchmark):
    """Benchmark nested control flow structures."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        # First while loop
        state = await thinkagain.while_loop(  # type: ignore[assignment]
            lambda s: s.value < 10,
            increment,
            state,
        )

        # Conditional
        state = await thinkagain.cond(  # type: ignore[assignment]
            lambda s: s.value < 20,
            increment,
            double,
            state,
        )

        # Second while loop
        state = await thinkagain.while_loop(  # type: ignore[assignment]
            lambda s: s.value < 50,
            increment,
            state,
        )

        return state

    result = await benchmark(pipeline, State(value=0))
    assert result.value == 50


@pytest.mark.asyncio
async def test_bench_complex_pipeline(benchmark):
    """Benchmark complex pipeline with multiple operations."""

    @thinkagain.node
    async def add(a: int, b: int) -> int:
        return a + b

    @thinkagain.jit
    async def pipeline(x: int) -> int:
        # Linear operations
        x = await add_one(x)
        x = await multiply(x, 2)

        # Scan over range
        items = [str(i) for i in range(10)]
        count, _ = await thinkagain.scan(process_item, 0, items)

        # Add scan result
        x = await add(x, count)

        # Final multiplication
        x = await multiply(x, 3)
        return x

    result = await benchmark(pipeline, 5)
    assert result > 0


@pytest.mark.asyncio
async def test_bench_static_kwargs_compilation(benchmark):
    """Benchmark compilation with static kwargs."""

    @thinkagain.jit(static_argnames=("scale",))
    async def pipeline(x: int, *, scale: int = 1) -> int:
        return await multiply(x, scale)

    # Benchmark one specific scale value
    result = await benchmark.pedantic(
        lambda: pipeline(2, scale=2),
        setup=lambda: _compiled_cache.clear(),
        rounds=3,
    )
    assert result == 4


# =============================================================================
# Traceable Container Benchmarks
# =============================================================================


@pytest.mark.asyncio
async def test_bench_frozen_dataclass(benchmark):
    """Benchmark with frozen dataclass containers."""

    @thinkagain.trace
    @dataclass(frozen=True)
    class FrozenState:
        value: int
        tag: str = ""

    @thinkagain.node
    async def process_frozen(state: FrozenState) -> FrozenState:
        return FrozenState(value=state.value + 1, tag=f"{state.tag}_processed")

    @thinkagain.jit
    async def pipeline(x: int) -> FrozenState:
        state = FrozenState(value=x)
        for _ in range(10):
            state = await process_frozen(state)
        return state

    result = await benchmark(pipeline, 1)
    assert result.value == 11


@pytest.mark.asyncio
async def test_bench_multiple_pipeline_executions(benchmark):
    """Benchmark multiple executions of the same pipeline."""

    @thinkagain.jit
    async def pipeline(x: int) -> int:
        x = await add_one(x)
        x = await multiply(x, 2)
        return x

    # Warm up
    await pipeline(1)

    # Benchmark repeated executions
    async def run_multiple():
        results = []
        for i in range(50):
            result = await pipeline(i)
            results.append(result)
        return results

    results = await benchmark(run_multiple)
    assert len(results) == 50

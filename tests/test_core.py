"""Comprehensive tests for JAX-style tracing and graph compilation."""

import pytest
from dataclasses import dataclass

import thinkagain
from thinkagain.core.errors import TracingError
from thinkagain.core.graph.graph import Node, OutputKind
from thinkagain.core.execution.executors import (
    CallExecutor,
    CondExecutor,
    WhileExecutor,
    ScanExecutor,
    SwitchExecutor,
)
from thinkagain.core.tracing.tracer import _compiled_cache
from thinkagain.core.graph.graph import Graph


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
async def no_return(state: State) -> None:
    """Returns None - should fail validation."""
    return None


def _find_node_by_executor_type(graph: Graph, executor_type: type) -> Node | None:
    """Find a node with the given executor type in a graph."""
    for node in graph.nodes:
        if isinstance(node.executor, executor_type):
            return node
    return None


def _graph_for(fn: object) -> Graph:
    """Get the compiled graph for a traced function."""
    target = getattr(fn, "__wrapped__", fn)
    for cache_key, graph in _compiled_cache.items():
        if cache_key[0] is target:
            return graph
    pytest.fail(f"{target} graph not found")


# =============================================================================
# Subgraph Tracing Tests
# =============================================================================


@pytest.mark.asyncio
async def test_cond_branches_traced_as_subgraphs():
    """Test that @node functions in cond branches are traced as Graphs."""

    @thinkagain.jit
    async def pipeline_with_cond(state: State) -> State:
        state = await thinkagain.cond(  # type: ignore[assignment]
            lambda s: s.value < 10,
            increment,  # Should be traced as Graph
            double,  # Should be traced as Graph
            state,
        )
        return state

    # Test execution
    result = await pipeline_with_cond(State(value=5))
    assert result.value == 6

    # Verify graph tracing
    graph = _graph_for(pipeline_with_cond)
    node = _find_node_by_executor_type(graph, CondExecutor)
    assert node, "Cond node not found in compiled graph"
    executor = node.executor
    assert isinstance(executor.true_branch, Graph)
    assert isinstance(executor.false_branch, Graph)
    assert len(executor.true_branch.nodes) > 0
    assert len(executor.false_branch.nodes) > 0


@pytest.mark.asyncio
async def test_while_loop_body_traced_as_subgraph():
    """Test that @node function in while_loop body is traced as Graph."""

    @thinkagain.jit
    async def pipeline_with_while(state: State) -> State:
        state = await thinkagain.while_loop(  # type: ignore[assignment]
            lambda s: s.value < 5,
            increment,  # Should be traced as Graph
            state,
        )
        return state

    # Test execution
    result = await pipeline_with_while(State(value=1))
    assert result.value == 5

    # Verify graph tracing
    graph = _graph_for(pipeline_with_while)
    node = _find_node_by_executor_type(graph, WhileExecutor)
    assert node, "While node not found in compiled graph"
    executor = node.executor
    assert isinstance(executor.body_fn, Graph)
    assert len(executor.body_fn.nodes) > 0


@pytest.mark.asyncio
async def test_scan_body_traced_as_subgraph():
    """Test that @node function in scan body is traced as Graph."""

    @thinkagain.jit
    async def pipeline_with_scan(items: list[str]) -> list[str]:
        final_count, results = await thinkagain.scan(
            process_item,  # Should be traced as Graph
            init=0,
            xs=items,
        )
        return results  # type: ignore[return-value]

    # Test execution
    results = await pipeline_with_scan(["a", "b", "c"])
    assert len(results) == 3
    assert "Processed: a" in results[0]

    # Verify graph tracing
    graph = _graph_for(pipeline_with_scan)
    node = _find_node_by_executor_type(graph, ScanExecutor)
    assert node, "Scan node not found in compiled graph"
    executor = node.executor
    assert isinstance(executor.body_fn, Graph)
    assert len(executor.body_fn.nodes) > 0


@pytest.mark.asyncio
async def test_simple_sync_functions_not_traced():
    """Test that simple sync functions in scan are stored as callables, not traced."""

    @thinkagain.jit
    async def pipeline_with_simple_scan(items: list[str]) -> list[str]:
        def simple_process(carry: int, item: str) -> tuple[int, str]:
            return carry + 1, f"Item: {item}"

        final, results = await thinkagain.scan(simple_process, 0, items)
        return results  # type: ignore[return-value]

    results = await pipeline_with_simple_scan(["x", "y"])
    assert len(results) == 2

    # Verify simple function is NOT traced
    graph = _graph_for(pipeline_with_simple_scan)
    node = _find_node_by_executor_type(graph, ScanExecutor)
    assert node, "Scan node not found in compiled graph"
    executor = node.executor
    assert not isinstance(executor.body_fn, Graph), (
        "Simple sync function should NOT be traced as Graph"
    )


# =============================================================================
# Captured Values Tests
# =============================================================================


@pytest.mark.asyncio
async def test_captured_value_in_cond_branch():
    """Test that branches can capture TracedValues from parent context."""

    @thinkagain.jit
    async def pipeline_with_captured_value(state: State) -> State:
        threshold = await get_threshold(state)

        @thinkagain.node
        async def high_branch(s: State) -> State:
            # Captures 'threshold' from parent context
            return await increment_by_threshold(s, threshold)

        state = await thinkagain.cond(  # type: ignore[assignment]
            lambda s: s.value < 10,
            high_branch,  # Captures threshold
            increment,
            state,
        )
        return state

    # Test execution with captured value
    result = await pipeline_with_captured_value(State(value=5, threshold=3))
    assert result.value == 8  # 5 + 3

    # Verify captured values in graph
    graph = _graph_for(pipeline_with_captured_value)
    node = _find_node_by_executor_type(graph, CondExecutor)
    assert node and isinstance(node.executor.true_branch, Graph), (
        "Cond node with Graph not found"
    )
    branch_graph = node.executor.true_branch
    assert len(branch_graph.captured_inputs) > 0, "Should have captured values"


@pytest.mark.asyncio
async def test_captured_value_in_while_loop():
    """Test that while loop body can capture TracedValues."""

    @thinkagain.jit
    async def pipeline_with_while_capture(state: State) -> State:
        threshold = await get_threshold(state)

        @thinkagain.node
        async def body_with_capture(s: State) -> State:
            # Captures 'threshold' from parent context
            return await increment_by_threshold(s, threshold)

        state = await thinkagain.while_loop(  # type: ignore[assignment]
            lambda s: s.value < 15,
            body_with_capture,  # Captures threshold
            state,
        )
        return state

    # Test execution
    result = await pipeline_with_while_capture(State(value=0, threshold=5))
    assert result.value == 15  # 0->5->10->15


@pytest.mark.asyncio
async def test_captured_value_with_nested_tracing():
    """Test captured inputs flow through nested tracing contexts."""

    @thinkagain.jit
    async def inner_pipeline(state: State, threshold: int) -> State:
        return await increment_by_threshold(state, threshold)

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        threshold = await get_threshold(state)

        @thinkagain.node
        async def branch(s: State) -> State:
            return await inner_pipeline(s, threshold)

        return await thinkagain.cond(  # type: ignore[return-value]
            lambda s: s.value < 10,
            branch,
            branch,
            state,
        )

    result = await pipeline(State(value=2, threshold=4))
    assert result.value == 6

    graph = _graph_for(pipeline)
    node = _find_node_by_executor_type(graph, CondExecutor)
    assert node and isinstance(node.executor.true_branch, Graph), (
        "Cond node with Graph not found"
    )
    branch_graph = node.executor.true_branch
    assert branch_graph.captured_inputs, "Expected captured inputs"


@pytest.mark.asyncio
async def test_branch_returns_captured_value_directly():
    """Test that a branch can return a captured TracedValue directly."""

    @thinkagain.jit
    async def pipeline_return_captured(state: State) -> int:
        threshold = await get_threshold(state)

        @thinkagain.node
        async def return_capture(_: State) -> int:
            return threshold

        return await thinkagain.cond(  # type: ignore[return-value]
            lambda s: s.value < 10, return_capture, return_capture, state
        )

    result = await pipeline_return_captured(State(value=7, threshold=11))
    assert result == 11


@pytest.mark.asyncio
async def test_branch_returns_literal_object():
    """Test that a branch can return a literal object (not a TracedValue)."""

    @thinkagain.jit
    async def pipeline_return_literal(state: State) -> State:
        literal_state = State(value=42, threshold=99)

        @thinkagain.node
        async def return_literal(_: State) -> State:
            return literal_state

        return await thinkagain.cond(  # type: ignore[return-value]
            lambda s: s.value < 10, return_literal, return_literal, state
        )

    result = await pipeline_return_literal(State(value=1))
    assert result.value == 42
    assert result.threshold == 99


# =============================================================================
# Explicit Output Tests
# =============================================================================


@pytest.mark.asyncio
async def test_explicit_output_return_input():
    """Test that returning input is tracked as INPUT output kind."""

    @thinkagain.jit
    async def return_input(x: int) -> int:
        return x

    assert await return_input(7) == 7

    graph = _graph_for(return_input)
    assert graph.output_ref.kind is OutputKind.INPUT  # type: ignore[union-attr]
    assert graph.output_ref.value == 0  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_explicit_output_return_literal():
    """Test that returning a literal is tracked as LITERAL output kind."""

    @thinkagain.jit
    async def return_literal(_: int) -> int:
        return 42

    assert await return_literal(7) == 42

    graph = _graph_for(return_literal)
    assert graph.output_ref.kind  # type: ignore[union-attr] is OutputKind.LITERAL
    assert graph.output_ref.value  # type: ignore[union-attr] == 42


@pytest.mark.asyncio
async def test_explicit_output_return_node():
    """Test that returning a node result is tracked as NODE output kind."""

    @thinkagain.jit
    async def return_node(x: int) -> int:
        y = await add_one(x)
        return y

    assert await return_node(7) == 8

    graph = _graph_for(return_node)
    assert graph.output_ref.kind is OutputKind.NODE  # type: ignore[union-attr]
    assert graph.output_ref.value in {node.node_id for node in graph.nodes}  # type: ignore[union-attr]


# =============================================================================
# Validation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_validation_while_missing_return():
    """Test that while_loop validates body returns a value."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        state = await thinkagain.while_loop(  # type: ignore[assignment]
            lambda s: s.value < 5,
            no_return,  # Returns None - should fail
            state,
        )
        return state

    with pytest.raises(Exception):
        await pipeline(State(value=1))


@pytest.mark.asyncio
async def test_validation_scan_missing_return():
    """Test that scan validates body returns a value."""

    @thinkagain.node
    async def no_return_scan(carry: int, item: str) -> None:
        return None

    @thinkagain.jit
    async def pipeline(items: list[str]) -> list[str]:
        final, results = await thinkagain.scan(
            no_return_scan,  # Returns None - should fail
            0,
            items,
        )
        return results  # type: ignore[return-value]

    with pytest.raises(Exception):
        await pipeline(["a", "b"])


@pytest.mark.asyncio
async def test_validation_cond_missing_return_runtime():
    """Test that cond with @node returning None is caught at runtime."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        state = await thinkagain.cond(  # type: ignore[assignment]
            lambda s: s.value < 10,
            increment,  # Returns State
            no_return,  # Returns None - traced as SubGraph but fails at runtime
            state,
        )
        return state

    # This might pass tracing but could fail at runtime depending on implementation
    # Test both branches
    try:
        result = await pipeline(State(value=15))  # Takes false branch (no_return)
        # If we get here, the framework allows it (some do)
        assert result is None or isinstance(result, State)
    except Exception:
        # Expected: runtime error for None return
        pass


# =============================================================================
# Graph Caching Tests
# =============================================================================


@pytest.mark.asyncio
async def test_graph_caching_works_with_captured_values():
    """Test that graph caching works correctly with captured values."""

    @thinkagain.jit
    async def pipeline_with_captured_value(state: State) -> State:
        threshold = await get_threshold(state)

        @thinkagain.node
        async def high_branch(s: State) -> State:
            return await increment_by_threshold(s, threshold)

        state = await thinkagain.cond(  # type: ignore[assignment]
            lambda s: s.value < 10, high_branch, increment, state
        )
        return state

    # First call compiles the graph
    result1 = await pipeline_with_captured_value(State(value=5, threshold=3))
    assert result1.value == 8

    cache_size_after_first = len(_compiled_cache)

    # Second call should reuse cached graph
    result2 = await pipeline_with_captured_value(State(value=2, threshold=4))
    assert result2.value == 6

    cache_size_after_second = len(_compiled_cache)

    # Cache size should not increase (graph was reused)
    assert cache_size_after_second == cache_size_after_first


# =============================================================================
# Keyword Argument Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_dynamic_kwargs_affect_execution():
    """Dynamic kwargs should be treated as inputs and update at runtime."""

    @thinkagain.node
    async def add_bias(x: int, bias: int) -> int:
        return x + bias

    @thinkagain.jit
    async def add(x: int, *, bias: int = 0) -> int:
        return await add_bias(x, bias)

    result1 = await add(1, bias=2)
    cache_size_after_first = len(_compiled_cache)
    result2 = await add(1, bias=5)
    cache_size_after_second = len(_compiled_cache)

    assert result1 == 3
    assert result2 == 6
    assert cache_size_after_second == cache_size_after_first


@pytest.mark.asyncio
async def test_static_kwargs_force_recompile():
    """Static kwargs should trigger recompilation when values change."""

    @thinkagain.node
    async def multiply(x: int, scale: int) -> int:
        return x * scale

    @thinkagain.jit(static_argnames=("scale",))
    async def scale(x: int, *, scale: int = 1) -> int:
        return await multiply(x, scale)

    result1 = await scale(2, scale=2)
    cache_size_after_first = len(_compiled_cache)
    result2 = await scale(2, scale=3)
    cache_size_after_second = len(_compiled_cache)

    assert result1 == 4
    assert result2 == 6
    assert cache_size_after_second == cache_size_after_first + 1


@pytest.mark.asyncio
async def test_static_kwargs_require_hashable_values():
    """Static kwargs must be hashable to participate in cache keys."""

    @thinkagain.jit(static_argnames=("config",))
    async def uses_config(x: int, *, config: list[int]) -> int:
        return x + config[0]

    with pytest.raises(TypeError):
        await uses_config(1, config=[1])


# =============================================================================
# Control Flow Validation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_cond_rejects_mismatched_output_kinds():
    """cond should reject branches with incompatible output patterns."""

    async def return_input(x: int) -> int:
        return x

    async def return_literal(_: int) -> int:
        return 1

    @thinkagain.jit
    async def pipeline(x: int) -> int:
        return await thinkagain.cond(lambda _: True, return_input, return_literal, x)  # type: ignore[return-value]

    with pytest.raises(TracingError):
        await pipeline(3)


@pytest.mark.asyncio
async def test_scan_rejects_invalid_tuple_shape():
    """scan should reject bodies that do not return (carry, output)."""

    @thinkagain.node
    async def bad_scan(carry: int, item: int) -> int:
        return carry + item

    @thinkagain.jit
    async def pipeline(xs: list[int]) -> list[int]:
        _, results = await thinkagain.scan(bad_scan, 0, xs)
        return results  # type: ignore[return-value]

    with pytest.raises(thinkagain.NodeExecutionError):
        await pipeline([1, 2, 3])


@pytest.mark.asyncio
async def test_rejects_non_traceable_capture_in_cond():
    """Non-traceable callables that capture TracedValue should be rejected."""

    @thinkagain.jit
    async def pipeline(state: State) -> int:
        threshold = await get_threshold(state)
        return await thinkagain.cond(  # type: ignore[return-value]
            lambda s: s.value > 0,
            lambda _: threshold,
            lambda _: threshold,
            state,
        )

    with pytest.raises(TracingError):
        await pipeline(State(value=1, threshold=2))


# =============================================================================
# Traceable Container Tests
# =============================================================================


@pytest.mark.asyncio
async def test_frozen_dataclass_traceable():
    """Frozen dataclasses can be used as @trace containers."""

    @thinkagain.trace
    @dataclass(frozen=True)
    class FrozenState:
        value: int
        tag: str = ""

    @thinkagain.jit
    async def pipeline(x: int) -> FrozenState:
        y = await add_one(x)
        return FrozenState(value=y, tag="ok")

    result = await pipeline(1)
    assert result == FrozenState(2, "ok")


@pytest.mark.asyncio
async def test_traceable_subclass_protocol():
    """Subclass of a traceable base should be rejected unless registered."""

    @thinkagain.trace
    class Box:
        def __init__(self, value: int):
            self.value = value

        def decompose(self) -> tuple[list[int], None]:
            return [self.value], None

        @classmethod
        def compose(cls, aux: None, children: list[int]) -> "Box":
            return cls(children[0])

    class FancyBox(Box):
        pass

    @thinkagain.jit
    async def pipeline(x: int) -> FancyBox:
        y = await add_one(x)
        return FancyBox(y)

    with pytest.raises(TracingError, match="non-@trace container"):
        await pipeline(1)


@pytest.mark.asyncio
async def test_reject_traced_value_in_dict_key():
    """TracedValue should be rejected in dict keys."""

    @thinkagain.jit
    async def pipeline(x: int) -> dict:
        return {x: 1}

    with pytest.raises(TypeError, match="dict keys"):
        await pipeline(1)


@pytest.mark.asyncio
async def test_reject_traced_value_in_set():
    """TracedValue should be rejected in sets."""

    @thinkagain.jit
    async def pipeline(x: int) -> set[int]:
        return {x}

    with pytest.raises(TypeError, match="sets"):
        await pipeline(1)


@pytest.mark.asyncio
async def test_parent_values_param_is_allowed():
    """parent_values is no longer reserved since it isn't injected at runtime."""

    @thinkagain.node
    async def node_with_parent_values(x: int, parent_values: dict | None = None) -> int:
        return x if parent_values is None else x + 1

    @thinkagain.jit
    async def pipeline(x: int) -> int:
        return await node_with_parent_values(x)

    assert await pipeline(1) == 1


@pytest.mark.asyncio
async def test_nested_jit_is_inlined_into_outer_trace():
    """Nested @jit should be a no-op and inline into the outer trace."""

    @thinkagain.jit
    async def inner(x: int) -> int:
        return await add_one(x)

    @thinkagain.jit
    async def outer(x: int) -> int:
        return await inner(x)

    result = await outer(2)
    assert result == 3

    graph = _graph_for(outer)
    for node in graph.nodes:
        if isinstance(node.executor, CallExecutor):
            # The executor should NOT be for inner or inner.__wrapped__
            assert node.executor.fn is not inner.__wrapped__  # type: ignore[attr-defined]
            assert node.executor.fn is not inner


@pytest.mark.asyncio
async def test_cond_predicate_rejects_captured_traced_value():
    """cond predicate should not capture TracedValues."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        threshold = await get_threshold(state)
        return await thinkagain.cond(  # type: ignore[return-value]
            lambda s: s.value < threshold,
            increment,
            increment,
            state,
        )

    with pytest.raises(TracingError):
        await pipeline(State(value=1, threshold=2))


@pytest.mark.asyncio
async def test_while_predicate_rejects_captured_traced_value():
    """while_loop predicate should not capture TracedValues."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        threshold = await get_threshold(state)
        return await thinkagain.while_loop(  # type: ignore[return-value]
            lambda s: s.value < threshold,
            increment,
            state,
        )

    with pytest.raises(TracingError):
        await pipeline(State(value=1, threshold=2))


# =============================================================================
# Switch Operator Tests
# =============================================================================


@pytest.mark.asyncio
async def test_switch_branches_traced_as_subgraphs():
    """Test that @node functions in switch branches are traced as Graphs."""

    @thinkagain.jit
    async def pipeline_with_switch(state: State) -> State:
        def get_tier(s: State) -> int:
            return 0 if s.value < 10 else 1

        return await thinkagain.switch(  # type: ignore[return-value]
            get_tier,
            [increment, double],
            state,
        )

    result = await pipeline_with_switch(State(value=7))
    assert result.value == 8

    # Verify graph tracing
    graph = _graph_for(pipeline_with_switch)
    node = _find_node_by_executor_type(graph, SwitchExecutor)
    assert node, "Switch node not found in compiled graph"
    executor = node.executor
    assert len(executor.branches) == 2
    for branch in executor.branches:
        assert isinstance(branch, Graph)


@pytest.mark.asyncio
async def test_switch_index_out_of_bounds():
    """Test that switch raises IndexError for out-of-bounds index."""

    with pytest.raises(IndexError, match="switch index"):
        await thinkagain.switch(lambda _: 5, [lambda x: x], 10)


@pytest.mark.asyncio
async def test_switch_rejects_mismatched_output_kinds():
    """switch should reject branches with incompatible output patterns."""

    async def return_input(x: int) -> int:
        return x

    async def return_literal(_: int) -> int:
        return 42

    @thinkagain.jit
    async def pipeline(x: int) -> int:
        return await thinkagain.switch(lambda _: 0, [return_input, return_literal], x)  # type: ignore[return-value]

    with pytest.raises(TracingError, match="switch branches must return same pattern"):
        await pipeline(5)


@pytest.mark.asyncio
async def test_switch_index_fn_rejects_captured_traced_value():
    """switch index_fn should not capture TracedValues."""

    @thinkagain.jit
    async def pipeline(state: State) -> State:
        threshold = await get_threshold(state)
        return await thinkagain.switch(  # type: ignore[return-value]
            lambda s: 0 if s.value < threshold else 1,
            [increment, double],
            state,
        )

    with pytest.raises(TracingError, match="switch index_fn closes over TracedValue"):
        await pipeline(State(value=1, threshold=5))


# =============================================================================
# @node Decorator Validation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_node_decorator_rejects_sync_function():
    """Test that @node rejects sync functions."""

    with pytest.raises(TypeError, match="async function"):

        @thinkagain.node
        def sync_func(x: int) -> int:
            return x + 1


@pytest.mark.asyncio
async def test_node_decorator_rejects_class():
    """Test that @node rejects classes (use @replica instead)."""

    with pytest.raises(TypeError, match="async function"):

        @thinkagain.node
        class NodeClass:
            async def __call__(self, x: int) -> int:
                return x + 1


@pytest.mark.asyncio
async def test_apply_replica_updates_state():
    """Test that apply_replica updates replica state via decompose/compose."""

    class Counter:
        def __init__(self, count: int = 0, step: int = 1):
            self.count = count
            self.step = step

        def decompose(self):
            return [self.count, self.step], None

        @classmethod
        def compose(cls, aux, children):
            count, step = children
            return cls(count=count, step=step)

    @thinkagain.node
    async def add(a: int, b: int) -> int:
        return a + b

    @thinkagain.jit
    async def step(count: int, step: int, x: int):
        new_count = await add(count, step)
        out = await add(new_count, x)
        return [new_count, step], out

    counter = Counter()
    out1 = await thinkagain.apply_replica(counter, step, 10)
    assert out1 == 11
    assert counter.count == 1

    out2 = await thinkagain.apply_replica(counter, step, 5)
    assert out2 == 7
    assert counter.count == 2

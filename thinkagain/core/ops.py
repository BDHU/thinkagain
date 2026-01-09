"""JAX-style control flow operators for @jit functions.

These operators trace control flow (conditionals, loops) into the computation graph.
"""

from typing import Callable, TypeVar

from .errors import TracingError
from .executors import (
    CallExecutor,
    CondExecutor,
    ScanExecutor,
    SwitchExecutor,
    WhileExecutor,
)
from .graph import Graph, TracedValue
from .runtime import maybe_await
from .tracing import (
    _captures_traced_value,
    _get_source_location,
    get_trace_context,
    is_traceable,
    trace_branch,
    validate_cond_branches,
    validate_scan_body,
    validate_switch_branches,
    validate_while_body,
)

T = TypeVar("T")
U = TypeVar("U")


def _check_no_capture(fn: Callable, ctx, name: str) -> None:
    """Ensure a predicate/condition function doesn't capture traced values."""
    if _captures_traced_value(fn, ctx):
        raise TracingError(
            f"{name} closes over TracedValue; predicates run at runtime "
            "and must not capture traced values."
        )


def _check_body_capture(fn: Callable, ctx, name: str) -> None:
    """Ensure a body function either is traceable or doesn't capture traced values."""
    if not is_traceable(fn) and _captures_traced_value(fn, ctx):
        raise TracingError(
            f"{name} closes over TracedValue but is not traceable; "
            "use async @node or async function instead."
        )


async def _maybe_trace(fn: Callable, input_count: int, ctx) -> Graph | Callable:
    """Trace a function into a graph if traceable, otherwise return as-is."""
    if is_traceable(fn):
        return await trace_branch(fn, input_count, ctx)
    return fn


# ---------------------------------------------------------------------------
# Control Flow Operators
# ---------------------------------------------------------------------------


async def cond(
    pred_fn: Callable[[T], bool],
    true_fn: Callable[[T], U],
    false_fn: Callable[[T], U],
    operand: T,
) -> U:
    """Conditional execution for computation graphs.

    Inside @jit: Traced as a graph node
    Outside @jit: Executes like normal if/else

    Args:
        pred_fn: Predicate function (sync or async)
        true_fn: Called if predicate is True
        false_fn: Called if predicate is False
        operand: Input passed to predicate and chosen branch

    Example:
        @jit
        async def pipeline(state):
            return await cond(
                lambda s: s.score > 0.5,
                high_quality_path,
                low_quality_path,
                state
            )
    """
    ctx = get_trace_context()

    if ctx is None:
        # Outside @jit - execute directly
        result = await maybe_await(pred_fn, operand)
        return await maybe_await(true_fn if result else false_fn, operand)

    # Inside @jit - validate and trace
    _check_no_capture(pred_fn, ctx, "cond pred_fn")
    _check_body_capture(true_fn, ctx, "cond true_fn")
    _check_body_capture(false_fn, ctx, "cond false_fn")

    true_branch = await _maybe_trace(true_fn, 1, ctx)
    false_branch = await _maybe_trace(false_fn, 1, ctx)
    validate_cond_branches(true_branch, false_branch)

    executor = CondExecutor(
        pred_fn=pred_fn,
        true_branch=true_branch,
        false_branch=false_branch,
    )
    node_id = ctx.add_node(executor, (operand,), {}, _get_source_location(pred_fn))
    return TracedValue(node_id, ctx)


async def while_loop(
    cond_fn: Callable[[T], bool],
    body_fn: Callable[[T], T],
    init: T,
) -> T:
    """While loop for computation graphs.

    Inside @jit: Traced as a graph node
    Outside @jit: Executes like normal while loop

    Args:
        cond_fn: Condition function (sync or async)
        body_fn: Body function that transforms state
        init: Initial state

    Returns:
        Final state after loop terminates

    Example:
        @jit
        async def retry(state):
            return await while_loop(
                lambda s: s.attempts < 3 and s.quality < 0.8,
                retry_step,
                state
            )
    """
    ctx = get_trace_context()

    if ctx is None:
        # Outside @jit - execute directly
        state = init
        while await maybe_await(cond_fn, state):
            state = await maybe_await(body_fn, state)
        return state

    # Inside @jit - validate and trace
    _check_no_capture(cond_fn, ctx, "while_loop cond_fn")
    _check_body_capture(body_fn, ctx, "while_loop body_fn")

    body = await _maybe_trace(body_fn, 1, ctx)
    validate_while_body(body)

    executor = WhileExecutor(cond_fn=cond_fn, body_fn=body)
    node_id = ctx.add_node(executor, (init,), {}, _get_source_location(cond_fn))
    return TracedValue(node_id, ctx)


async def scan(
    fn: Callable[[T, U], tuple[T, U]],
    init: T,
    xs: list[U],
) -> tuple[T, list[U]]:
    """Scan over a sequence (like reduce but returns all outputs).

    Most efficient control flow since iteration count is known at trace time.

    Args:
        fn: Function (carry, x) -> (new_carry, output)
        init: Initial carry value
        xs: Sequence to iterate over

    Returns:
        (final_carry, list_of_outputs)

    Example:
        @jit
        async def batch(items):
            def process(count, item):
                return count + 1, f"#{count + 1}: {item}"
            return await scan(process, 0, items)
    """
    ctx = get_trace_context()

    if ctx is None:
        # Outside @jit - execute directly
        carry = init
        outputs = []
        for x in xs:
            carry, out = await maybe_await(fn, carry, x)
            outputs.append(out)
        return carry, outputs

    # Inside @jit - validate and trace
    _check_body_capture(fn, ctx, "scan fn")

    body = await _maybe_trace(fn, 2, ctx)
    validate_scan_body(body)

    executor = ScanExecutor(body_fn=body)
    scan_id = ctx.add_node(executor, (init, xs), {}, _get_source_location(fn))

    # Create extraction nodes for the tuple elements
    extract_carry = CallExecutor(fn=lambda t: t[0])
    extract_outputs = CallExecutor(fn=lambda t: t[1])

    carry_id = ctx.add_node(extract_carry, (TracedValue(scan_id, ctx),), {})
    outputs_id = ctx.add_node(extract_outputs, (TracedValue(scan_id, ctx),), {})

    return TracedValue(carry_id, ctx), TracedValue(outputs_id, ctx)


async def switch(
    index_fn: Callable[[T], int],
    branches: list[Callable[[T], U]],
    operand: T,
) -> U:
    """Multi-way branch selection for computation graphs.

    Inside @jit: Traced as a graph node
    Outside @jit: Executes like match/case statement

    Args:
        index_fn: Function that returns branch index (0-based)
        branches: List of branch functions to choose from
        operand: Input passed to index function and chosen branch

    Returns:
        Result from the selected branch

    Raises:
        IndexError: If index is out of bounds
        TracingError: If branches have incompatible signatures

    Example:
        @jit
        async def route(request):
            return await switch(
                lambda r: {"GET": 0, "POST": 1, "DELETE": 2}.get(r.method, 3),
                [handle_get, handle_post, handle_delete, handle_default],
                request
            )

    Note:
        More efficient than nested cond() calls for 3+ branches.
        Similar to jax.lax.switch() in JAX.
    """
    if not branches:
        raise ValueError("switch requires at least one branch")

    ctx = get_trace_context()

    if ctx is None:
        # Outside @jit - execute directly
        index = await maybe_await(index_fn, operand)
        if not isinstance(index, int) or index < 0 or index >= len(branches):
            raise IndexError(f"switch index {index} out of range [0, {len(branches)})")
        return await maybe_await(branches[index], operand)

    # Inside @jit - validate and trace
    _check_no_capture(index_fn, ctx, "switch index_fn")

    # Validate and trace all branches
    traced_branches = []
    for i, branch_fn in enumerate(branches):
        _check_body_capture(branch_fn, ctx, f"switch branch {i}")
        traced_branch = await _maybe_trace(branch_fn, 1, ctx)
        traced_branches.append(traced_branch)

    validate_switch_branches(traced_branches)

    executor = SwitchExecutor(index_fn=index_fn, branches=tuple(traced_branches))
    node_id = ctx.add_node(executor, (operand,), {}, _get_source_location(index_fn))
    return TracedValue(node_id, ctx)

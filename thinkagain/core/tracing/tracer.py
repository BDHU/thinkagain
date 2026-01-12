"""JAX-style tracing infrastructure for graph capture.

Functions decorated with @jit are traced once to build a computation graph,
then executed efficiently on subsequent calls.
"""

from __future__ import annotations

import functools
import inspect
from collections import OrderedDict
from typing import Any, Callable, TypeVar, cast

from ..errors import TracingError
from ..graph.graph import Graph, OutputKind, OutputRef, TracedValue
from ..graph.literal_refs import normalize_traced_literal
from ..services import register_service_bindings
from .context import TraceContext, _trace_ctx_var
from .utils import captures_traced_value, contains_traced_value, get_source_location

_captures_traced_value = captures_traced_value
_get_source_location = get_source_location

T = TypeVar("T")


def _callable_name(fn: Callable) -> str:
    return getattr(fn, "__name__", fn.__class__.__name__)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

DEFAULT_CACHE_SIZE = 4096


class GraphCache:
    """LRU cache for compiled graphs."""

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        self._cache: OrderedDict[tuple, Graph] = OrderedDict()
        self._max_size = max_size

    def get(self, key: tuple) -> Graph | None:
        return self._cache.get(key)

    def __contains__(self, key: tuple) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self):
        return iter(self._cache)

    def items(self):
        return self._cache.items()

    def put(self, key: tuple, graph: Graph) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = graph
        self._evict()

    def clear(self) -> None:
        self._cache.clear()

    def set_max_size(self, max_size: int) -> None:
        self._max_size = max_size
        self._evict()

    def info(self) -> dict[str, Any]:
        return {"cached_graphs": len(self._cache), "max_size": self._max_size}

    def _evict(self) -> None:
        if self._max_size >= 0:
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)


_compiled_cache = GraphCache()


def clear_compiled_cache() -> None:
    """Clear the compiled graph cache."""
    _compiled_cache.clear()


def set_cache_size(max_size: int) -> None:
    """Set maximum cache size (-1 for unlimited)."""
    _compiled_cache.set_max_size(max_size)


def get_cache_info() -> dict[str, Any]:
    """Get cache statistics."""
    return _compiled_cache.info()


# ---------------------------------------------------------------------------
# Graph Tracing
# ---------------------------------------------------------------------------


def _prepare_traced_inputs(
    input_count: int,
    kwargs: dict,
    static_argnames: set[str],
    ctx: TraceContext,
) -> tuple[list[TracedValue], list[TracedValue], dict[str, Any], tuple[str, ...]]:
    """Create traced inputs and kwargs for a trace session."""
    dynamic_kw_names = tuple(sorted(k for k in kwargs if k not in static_argnames))

    pos_inputs = [TracedValue(i, ctx) for i in range(input_count)]
    ctx.input_values = {v: i for i, v in enumerate(pos_inputs)}

    kw_inputs: list[TracedValue] = []
    traced_kwargs: dict[str, Any] = {}
    for name in sorted(kwargs):
        if name in static_argnames:
            traced_kwargs[name] = kwargs[name]
        else:
            idx = input_count + len(kw_inputs)
            tv = TracedValue(idx, ctx)
            kw_inputs.append(tv)
            ctx.input_values[tv] = idx
            traced_kwargs[name] = tv

    ctx.next_capture_index = input_count + len(kw_inputs)
    return pos_inputs, pos_inputs + kw_inputs, traced_kwargs, dynamic_kw_names


def _get_trace_fn(fn: Callable, parent_ctx: TraceContext | None) -> Callable:
    """Get the actual function to trace (unwrap @node if needed)."""
    node_fn = getattr(fn, "_node_fn", None)
    if callable(node_fn) and captures_traced_value(node_fn, parent_ctx):
        return node_fn
    # For callable instances with async __call__, trace the __call__ method if it captures values
    call = getattr(fn, "__call__", None)
    if (
        call is not None
        and inspect.iscoroutinefunction(call)
        and captures_traced_value(call, parent_ctx)
    ):
        return call
    return fn


def _make_output_ref(
    result: Any,
    inputs: list[TracedValue],
    ctx: TraceContext,
    allow_parent: bool = False,
) -> OutputRef:
    """Create OutputRef from trace result."""
    if isinstance(result, TracedValue):
        if result in inputs:
            return OutputRef(OutputKind.INPUT, inputs.index(result))
        if result.trace_ctx is ctx:
            return OutputRef(OutputKind.NODE, result.node_id)
        if allow_parent and ctx.parent_ctx and result.trace_ctx is ctx.parent_ctx:
            idx = ctx._register_capture(result)
            return OutputRef(OutputKind.INPUT, idx)
        raise TracingError("TracedValue from wrong context.")
    # Normalize any TracedValues inside literal containers
    normalized = normalize_traced_literal(result, inputs, ctx)
    if contains_traced_value(normalized, ctx, depth=4):
        raise TracingError(
            "TracedValue is hidden inside a non-@trace container. "
            "Register the type with @trace or refactor to use built-in containers."
        )
    return OutputRef(OutputKind.LITERAL, normalized)


async def _trace_fn(
    fn: Callable,
    input_count: int,
    *,
    parent_ctx: TraceContext | None = None,
    kwargs: dict | None = None,
    allow_parent: bool = False,
    static_argnames: set[str] | None = None,
) -> Graph:
    """Trace a function into a graph."""
    ctx = TraceContext(parent_ctx=parent_ctx, next_capture_index=input_count)
    token = _trace_ctx_var.set(ctx)
    kwargs = kwargs or {}
    static_argnames = static_argnames or set()
    try:
        trace_fn = _get_trace_fn(fn, parent_ctx)

        pos_inputs, all_inputs, traced_kwargs, dynamic_kw_names = (
            _prepare_traced_inputs(input_count, kwargs, static_argnames, ctx)
        )

        result = await trace_fn(*pos_inputs, **traced_kwargs)
        output_ref = _make_output_ref(result, all_inputs, ctx, allow_parent)

        # Total input count includes: positional args + dynamic kwargs + captured values + resources
        total_input_count = (
            input_count
            + (len(all_inputs) - input_count)
            + len(ctx.captured_inputs)
            + len(ctx.resource_list)
        )

        return Graph(
            nodes=ctx.nodes,
            input_count=total_input_count,
            output_ref=output_ref,
            captured_inputs=ctx.captured_inputs,
            resource_list=ctx.resource_list,
            dynamic_kw_names=dynamic_kw_names,
        )
    finally:
        _trace_ctx_var.reset(token)


async def trace_branch(
    fn: Callable, input_count: int = 1, parent_ctx: TraceContext | None = None
) -> Graph:
    """Trace a control flow branch into a graph."""
    return await _trace_fn(fn, input_count, parent_ctx=parent_ctx, allow_parent=True)


# ---------------------------------------------------------------------------
# Cache Key
# ---------------------------------------------------------------------------


def _cache_key(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    static_argnames: set[str],
    graph: "Graph | None" = None,
) -> tuple:
    """Compute cache key for compilation.

    Args:
        fn: Function being compiled
        args: Positional arguments
        kwargs: Keyword arguments
        static_argnames: Set of kwarg names that trigger recompilation
        graph: Optional graph (after tracing) to include resources

    Returns:
        Cache key tuple
    """
    arg_types = tuple(type(a) for a in args)
    kw_sig = []
    for k in sorted(kwargs):
        v = kwargs[k]
        if k in static_argnames:
            try:
                hash(v)
            except TypeError as e:
                raise TypeError(f"static kwarg '{k}' must be hashable") from e
            kw_sig.append((k, type(v), v))
        else:
            kw_sig.append((k, type(v)))

    # Include resources if graph is available (after tracing)
    resource_key = ()
    if graph is not None:
        resource_key = tuple(hash(r) for r in graph.resource_list)

    return (fn, arg_types, tuple(kw_sig), tuple(sorted(static_argnames)), resource_key)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def jit(
    fn: Callable[..., T] | None = None,
    *,
    static_argnames: list[str] | tuple[str, ...] | set[str] | None = None,
) -> Callable[..., T]:
    """Decorator to create a JIT compilation boundary.

    The decorated async function is traced once per unique argument signature,
    then the compiled graph is reused for subsequent calls.

    Args:
        fn: Async function to compile
        static_argnames: Kwargs that trigger recompilation when changed

    Example:
        @jit
        async def pipeline(state):
            return await process(state)
    """
    if fn is None:
        return lambda f: jit(f, static_argnames=static_argnames)

    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"@jit requires async function, got {_callable_name(fn)}")

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        static = set(static_argnames or [])

        # Nested @jit is a no-op (inline into parent trace)
        if _trace_ctx_var.get() is not None:
            return await fn(*args, **kwargs)

        # Get service provider from mesh if available
        from ...distributed import get_current_mesh

        mesh = get_current_mesh()
        service_provider = mesh.get_service_provider() if mesh else None

        # Check cache (without graph first)
        key_base = _cache_key(fn, args, kwargs, static, graph=None)
        graph = _compiled_cache.get(key_base)
        if graph is not None:
            bound_args = graph.bind_inputs(args, kwargs)
            return await graph.execute(
                *bound_args,
                service_provider=service_provider,
            )

        # Trace and cache
        graph = await _trace_fn(fn, len(args), kwargs=kwargs, static_argnames=static)

        # Cache with full key including resources
        key_full = _cache_key(fn, args, kwargs, static, graph=graph)
        _compiled_cache.put(key_full, graph)

        bound_args = graph.bind_inputs(args, kwargs)
        return await graph.execute(
            *bound_args,
            service_provider=service_provider,
        )

    wrapper._is_jit = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def node(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator to mark an async function as a traceable node.

    Inside @jit: Returns TracedValue, adds to computation graph
    Outside @jit: Executes normally

    Only works with async functions. For stateful objects, use @replica instead.

    Example:
        @node
        async def process(state, factor: int):
            return replace(state, value=state.value * factor)
    """
    from ..execution.executors import CallExecutor

    # Only accept async functions
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"@node requires an async function, got {_callable_name(fn)}. "
            f"For stateful objects, use @replica instead."
        )

    # Wrap in tracing logic
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        ctx = _trace_ctx_var.get()
        if ctx is not None:
            register_service_bindings(ctx, wrapper)

            # Use wrapper as fn so that attributes like _distribution_config are visible
            # to execution hooks (e.g., distributed execution hook)
            executor = CallExecutor(fn=wrapper)
            node_id = ctx.add_node(executor, args, kwargs, get_source_location(fn))
            return TracedValue(node_id, ctx)
        return await fn(*args, **kwargs)

    setattr(wrapper, "_is_node", True)
    setattr(wrapper, "_node_fn", fn)
    if hasattr(fn, "_service_bindings"):
        setattr(wrapper, "_service_bindings", fn._service_bindings)
    return cast(Callable[..., T], wrapper)

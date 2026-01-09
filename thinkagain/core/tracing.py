"""JAX-style tracing infrastructure for graph capture.

Functions decorated with @jit are traced once to build a computation graph,
then executed efficiently on subsequent calls.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from .errors import TracingError
from .graph import (
    Graph,
    InputRef,
    Node,
    NodeRef,
    OutputKind,
    OutputRef,
    TracedValue,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

DEFAULT_CACHE_SIZE = 4096

_compiled_cache: OrderedDict[tuple, Graph] = OrderedDict()
_cache_max_size: int = DEFAULT_CACHE_SIZE


def _cache_put(key: tuple, graph: Graph) -> None:
    """Add a graph to the cache with LRU eviction."""
    if key in _compiled_cache:
        _compiled_cache.move_to_end(key)
    _compiled_cache[key] = graph
    if _cache_max_size >= 0:
        while len(_compiled_cache) > _cache_max_size:
            _compiled_cache.popitem(last=False)


def clear_compiled_cache() -> None:
    """Clear the compiled graph cache."""
    _compiled_cache.clear()


def set_cache_size(max_size: int) -> None:
    """Set maximum cache size (-1 for unlimited)."""
    global _cache_max_size
    _cache_max_size = max_size
    if _cache_max_size >= 0:
        while len(_compiled_cache) > _cache_max_size:
            _compiled_cache.popitem(last=False)


def get_cache_info() -> dict[str, Any]:
    """Get cache statistics."""
    return {"cached_graphs": len(_compiled_cache), "max_size": _cache_max_size}


# ---------------------------------------------------------------------------
# Trace Context
# ---------------------------------------------------------------------------

_trace_ctx_var: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "trace_context", default=None
)


def get_trace_context() -> TraceContext | None:
    """Get the current trace context, or None if not tracing."""
    return _trace_ctx_var.get()


def is_tracing() -> bool:
    """Check if we're currently inside a tracing context."""
    return _trace_ctx_var.get() is not None


# ---------------------------------------------------------------------------
# Tracing Plugin Registry
# ---------------------------------------------------------------------------

# Plugin system to decouple core tracing from distributed modules
_tracing_plugins: list[Callable[[TraceContext], Any]] = []


def register_tracing_plugin(factory: Callable[[TraceContext], Any]) -> None:
    """Register a plugin factory that creates tracing hooks.

    Args:
        factory: Callable that takes a TraceContext and returns a hook object
                with record_call() and get_resource_index() methods (TracingHook protocol)

    Example:
        register_tracing_plugin(lambda ctx: MyTracingHook(ctx))
    """
    _tracing_plugins.append(factory)


def unregister_all_tracing_plugins() -> None:
    """Clear all registered tracing plugins."""
    _tracing_plugins.clear()


@dataclass
class TraceContext:
    """Context for capturing computation graph during tracing."""

    nodes: list[Node] = field(default_factory=list)
    node_counter: int = 0
    parent_ctx: TraceContext | None = None
    next_capture_index: int = 0
    captured_inputs: dict[int, int] = field(default_factory=dict)
    input_values: dict[TracedValue, int] = field(default_factory=dict)
    # Resource tracking (discovered during tracing) - e.g., service handles
    resources: dict[int, int] = field(
        default_factory=dict
    )  # id(resource) -> input_index
    resource_list: list = field(default_factory=list)  # Ordered list of resources

    def _normalize(self, value: Any) -> Any:
        """Convert TracedValue to appropriate reference type."""
        if not isinstance(value, TracedValue):
            return value
        if value in self.input_values:
            return InputRef(self.input_values[value])
        if value.trace_ctx is self:
            return NodeRef(value.node_id)
        if self.parent_ctx and value.trace_ctx is self.parent_ctx:
            return InputRef(self._register_capture(value))
        raise TracingError("TracedValue from unrelated trace context.")

    def _normalize_many(self, values: tuple | dict) -> tuple | dict:
        """Normalize a collection of values."""
        if isinstance(values, dict):
            return {k: self._normalize(v) for k, v in values.items()}
        return tuple(self._normalize(v) for v in values)

    def _register_capture(self, value: TracedValue) -> int:
        """Register a captured value from parent context."""
        if value.node_id not in self.captured_inputs:
            self.captured_inputs[value.node_id] = self.next_capture_index
            self.next_capture_index += 1
        return self.captured_inputs[value.node_id]

    def _next_id(self) -> int:
        """Get next node ID."""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def add_node(
        self,
        executor: Any,
        args: tuple,
        kwargs: dict,
        source_location: str | None = None,
    ) -> int:
        """Add a node to the graph with the given executor.

        This is the universal method for adding any type of node. The executor
        determines how the node will be executed.

        Args:
            executor: NodeExecutor implementation (e.g., CallExecutor, CondExecutor)
            args: Positional arguments (will be normalized)
            kwargs: Keyword arguments (will be normalized)
            source_location: Optional source location for debugging

        Returns:
            The node ID of the added node
        """
        node = Node(
            node_id=self._next_id(),
            args=self._normalize_many(args),
            kwargs=self._normalize_many(kwargs),
            executor=executor,
            source_location=source_location,
        )
        self.nodes.append(node)
        return node.node_id

    def get_resource_index(self, resource: Any) -> int:
        """Get or register input index for a resource (e.g., service handle).

        Args:
            resource: Resource instance to track

        Returns:
            Input index where this resource will be passed
        """
        resource_id = id(resource)

        if resource_id not in self.resources:
            # First time seeing this resource - register it
            resource_idx = self.next_capture_index
            self.resources[resource_id] = resource_idx
            self.resource_list.append(resource)
            self.next_capture_index += 1

        return self.resources[resource_id]


# ---------------------------------------------------------------------------
# Tracing Utilities
# ---------------------------------------------------------------------------


def _get_source_location(obj: Any) -> str | None:
    """Get source location of an object for debugging."""
    if obj is None or isinstance(obj, Graph):
        return None
    # For executors, try to get the location of the underlying function
    if hasattr(obj, "fn"):
        obj = obj.fn
    elif hasattr(obj, "pred_fn"):
        obj = obj.pred_fn
    try:
        return f"{inspect.getfile(obj)}:{inspect.getsourcelines(obj)[1]}"
    except (OSError, TypeError):
        return None


def _contains_traced_value(
    value: Any, ctx: TraceContext, *, _seen: set[int], _depth: int
) -> bool:
    if isinstance(value, TracedValue) and value.trace_ctx is ctx:
        return True
    if _depth <= 0:
        return False
    obj_id = id(value)
    if obj_id in _seen:
        return False
    _seen.add(obj_id)

    if isinstance(value, dict):
        return any(
            _contains_traced_value(v, ctx, _seen=_seen, _depth=_depth - 1)
            for v in value.values()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(
            _contains_traced_value(v, ctx, _seen=_seen, _depth=_depth - 1)
            for v in value
        )
    if isinstance(value, functools.partial):
        if _contains_traced_value(value.args, ctx, _seen=_seen, _depth=_depth - 1):
            return True
        if _contains_traced_value(
            value.keywords or {}, ctx, _seen=_seen, _depth=_depth - 1
        ):
            return True
    if hasattr(value, "__dict__"):
        return _contains_traced_value(vars(value), ctx, _seen=_seen, _depth=_depth - 1)
    return False


def _captures_traced_value(fn: Callable, ctx: TraceContext | None) -> bool:
    """Check if function captures TracedValue from the given context."""
    if not ctx:
        return False

    _seen: set[int] = set()

    if isinstance(fn, functools.partial):
        if _contains_traced_value(fn, ctx, _seen=_seen, _depth=2):
            return True

    defaults = getattr(fn, "__defaults__", None) or ()
    if _contains_traced_value(defaults, ctx, _seen=_seen, _depth=2):
        return True
    kwdefaults = getattr(fn, "__kwdefaults__", None) or {}
    if _contains_traced_value(kwdefaults, ctx, _seen=_seen, _depth=2):
        return True

    closure = getattr(fn, "__closure__", None)
    if not closure:
        return False
    for cell in closure:
        try:
            val = cell.cell_contents
            if _contains_traced_value(val, ctx, _seen=_seen, _depth=2):
                return True
        except ValueError:
            pass
    return False


def is_traceable(fn: Callable) -> bool:
    """Check if a function can be traced (async function or @node decorated)."""
    return (
        inspect.iscoroutinefunction(fn)
        or hasattr(fn, "_is_node")
        or inspect.iscoroutinefunction(getattr(fn, "__call__", None))
    )


# ---------------------------------------------------------------------------
# Graph Tracing
# ---------------------------------------------------------------------------


def _get_trace_fn(fn: Callable, parent_ctx: TraceContext | None) -> Callable:
    """Get the actual function to trace (unwrap @node if needed)."""
    if hasattr(fn, "_node_fn") and _captures_traced_value(fn._node_fn, parent_ctx):
        return fn._node_fn
    # For @node decorated classes, trace the __call__ method if it captures values
    if inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    ) and _captures_traced_value(fn.__call__, parent_ctx):
        return fn.__call__
    return fn


def _normalize_literal(
    value: Any,
    inputs: list[TracedValue],
    ctx: TraceContext,
) -> Any:
    """Recursively normalize TracedValues in a literal value to refs.

    This handles cases where a @jit function returns a literal container
    (dict, list, tuple) that contains TracedValue objects as elements.
    """
    if isinstance(value, TracedValue):
        # Normalize TracedValue to InputRef or NodeRef
        if value in inputs:
            return InputRef(inputs.index(value))
        if value.trace_ctx is ctx:
            return NodeRef(value.node_id)
        raise TracingError("TracedValue from wrong context in literal.")
    elif isinstance(value, dict):
        # Recursively normalize dict values
        return {k: _normalize_literal(v, inputs, ctx) for k, v in value.items()}
    elif isinstance(value, list):
        # Recursively normalize list elements
        return [_normalize_literal(v, inputs, ctx) for v in value]
    elif isinstance(value, tuple):
        # Recursively normalize tuple elements
        return tuple(_normalize_literal(v, inputs, ctx) for v in value)
    else:
        # Primitive value, return as-is
        return value


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
    normalized = _normalize_literal(result, inputs, ctx)
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

    # Register replica hooks for tracing via plugin system
    from ..core.replica import register_replica_hook, unregister_replica_hook

    replica_hook_registered = False
    registered_hooks = []

    # Instantiate all registered tracing plugins
    for plugin_factory in _tracing_plugins:
        hook = plugin_factory(ctx)
        if hook is not None:  # Plugin may return None if not applicable
            register_replica_hook(hook)
            registered_hooks.append(hook)
            replica_hook_registered = True

    try:
        trace_fn = _get_trace_fn(fn, parent_ctx)

        # Create traced inputs for positional args
        pos_inputs = [TracedValue(i, ctx) for i in range(input_count)]
        ctx.input_values = {v: i for i, v in enumerate(pos_inputs)}

        # Create traced inputs for dynamic kwargs
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
        all_inputs = pos_inputs + kw_inputs

        result = await trace_fn(*pos_inputs, **traced_kwargs)
        output_ref = _make_output_ref(result, all_inputs, ctx, allow_parent)

        # Total input count includes: positional args + dynamic kwargs + captured values + resources
        total_input_count = (
            input_count
            + len(kw_inputs)
            + len(ctx.captured_inputs)
            + len(ctx.resource_list)
        )

        return Graph(
            nodes=ctx.nodes,
            input_count=total_input_count,
            output_ref=output_ref,
            captured_inputs=ctx.captured_inputs,
            resource_list=ctx.resource_list,
        )
    finally:
        _trace_ctx_var.reset(token)
        # Unregister replica hook if we registered it
        if replica_hook_registered:
            unregister_replica_hook()


async def trace_branch(
    fn: Callable, input_count: int = 1, parent_ctx: TraceContext | None = None
) -> Graph:
    """Trace a control flow branch into a graph."""
    return await _trace_fn(fn, input_count, parent_ctx=parent_ctx, allow_parent=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_graph(graph: Graph | Callable, context: str) -> None:
    """Validate a graph or callable for control flow."""
    if not isinstance(graph, Graph):
        return
    if graph.output_ref is None:
        raise TracingError(f"{context} must return a value.")
    if graph.output_ref.kind is OutputKind.LITERAL and graph.output_ref.value is None:
        raise TracingError(f"{context} returns None.")


def validate_cond_branches(
    true_branch: Graph | Callable, false_branch: Graph | Callable
) -> None:
    """Validate cond branches have compatible outputs."""
    _validate_graph(true_branch, "cond true branch")
    _validate_graph(false_branch, "cond false branch")
    if isinstance(true_branch, Graph) and isinstance(false_branch, Graph):
        if true_branch.output_ref.kind is not false_branch.output_ref.kind:
            raise TracingError(
                f"cond branches must return same pattern: "
                f"true={true_branch.output_ref.kind.value}, "
                f"false={false_branch.output_ref.kind.value}"
            )


def validate_while_body(body: Graph | Callable) -> None:
    """Validate while_loop body."""
    _validate_graph(body, "while_loop body")


def validate_scan_body(body: Graph | Callable) -> None:
    """Validate scan body."""
    _validate_graph(body, "scan body")
    if isinstance(body, Graph) and body.output_ref.kind is OutputKind.LITERAL:
        val = body.output_ref.value
        if not isinstance(val, tuple) or len(val) != 2:
            raise TracingError(f"scan body must return (carry, output), got {val!r}")


def validate_switch_branches(branches: list[Graph | Callable]) -> None:
    """Validate switch branches have compatible outputs."""
    if not branches:
        raise TracingError("switch must have at least one branch")

    for i, branch in enumerate(branches):
        _validate_graph(branch, f"switch branch {i}")

    # Check all branches return same output pattern
    graph_branches = [b for b in branches if isinstance(b, Graph)]
    if len(graph_branches) > 1:
        first_kind = graph_branches[0].output_ref.kind
        for i, branch in enumerate(graph_branches[1:], start=1):
            if branch.output_ref.kind is not first_kind:
                raise TracingError(
                    f"switch branches must return same pattern: "
                    f"branch 0={first_kind.value}, "
                    f"branch {i}={branch.output_ref.kind.value}"
                )


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
            state = await step1(state)
            state = await cond(pred, branch_a, branch_b, state)
            return state
    """
    if fn is None:
        return lambda f: jit(f, static_argnames=static_argnames)

    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"@jit requires async function, got {fn.__name__}")

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        static = set(static_argnames or [])

        # Nested @jit is a no-op (inline into parent trace)
        if _trace_ctx_var.get() is not None:
            return await fn(*args, **kwargs)

        # Get service provider from mesh if available
        from ..distributed import get_current_mesh

        mesh = get_current_mesh()
        service_provider = mesh.get_service_provider() if mesh else None

        # Check cache (without graph first)
        key_base = _cache_key(fn, args, kwargs, static, graph=None)
        if key_base in _compiled_cache:
            graph = _compiled_cache[key_base]
            dynamic_kw = [kwargs[k] for k in sorted(kwargs) if k not in static]
            # Resources are appended after dynamic kwargs
            return await graph.execute(
                *args,
                *dynamic_kw,
                *graph.resource_list,
                service_provider=service_provider,
            )

        # Trace and cache
        graph = await _trace_fn(fn, len(args), kwargs=kwargs, static_argnames=static)

        # Cache with full key including resources
        key_full = _cache_key(fn, args, kwargs, static, graph=graph)
        _cache_put(key_full, graph)

        dynamic_kw = [kwargs[k] for k in sorted(kwargs) if k not in static]
        # Resources are appended after dynamic kwargs
        return await graph.execute(
            *args,
            *dynamic_kw,
            *graph.resource_list,
            service_provider=service_provider,
        )

    wrapper._is_jit = True
    return wrapper


def node(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator to mark an async function or class as a traceable node.

    Inside @jit: Returns TracedValue, adds to computation graph
    Outside @jit: Executes normally

    Works with:
    - Async functions
    - Classes with async __call__ method
    - Any callable that returns an awaitable

    Examples:
        @node
        async def retrieve_docs(state):
            docs = await db.search(state.query)
            return replace(state, documents=docs)

        @node
        class LLMNode:
            def __init__(self, model: str):
                self.model = model

            async def __call__(self, state):
                response = await llm.generate(self.model, state.query)
                return replace(state, answer=response)
    """
    from .executors import CallExecutor

    # Check if it's an async function or a class/callable with async __call__
    is_async_fn = inspect.iscoroutinefunction(fn)
    is_async_callable = inspect.iscoroutinefunction(getattr(fn, "__call__", None))

    if not (is_async_fn or is_async_callable):
        raise TypeError(
            f"@node requires async function or class with async __call__, "
            f"got {fn.__name__}"
        )

    # If it's a class with async __call__, wrap the __call__ method
    if inspect.isclass(fn):
        original_call = fn.__call__

        async def wrapped_call(self, *args, **kwargs):
            ctx = _trace_ctx_var.get()
            if ctx is not None:
                # Add the instance (self) as the callable node
                executor = CallExecutor(fn=self)
                node_id = ctx.add_node(
                    executor, args, kwargs, _get_source_location(self)
                )
                return TracedValue(node_id, ctx)
            return await original_call(self, *args, **kwargs)

        fn.__call__ = wrapped_call
        fn._is_node = True
        fn._node_fn = fn
        return fn

    # For async functions, wrap in tracing logic
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        ctx = _trace_ctx_var.get()
        if ctx is not None:
            # Use wrapper as fn so that attributes like _distribution_config are visible
            # to execution hooks (e.g., distributed execution hook)
            executor = CallExecutor(fn=wrapper)
            node_id = ctx.add_node(executor, args, kwargs, _get_source_location(fn))
            return TracedValue(node_id, ctx)
        return await fn(*args, **kwargs)

    wrapper._is_node = True
    wrapper._node_fn = fn
    return wrapper

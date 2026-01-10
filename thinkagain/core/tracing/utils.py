"""Helpers for tracing inspection and introspection."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from ..graph.graph import Graph, TracedValue
from .context import TraceContext


def get_source_location(obj: Any) -> str | None:
    """Get source location of an object for debugging."""
    if obj is None or isinstance(obj, Graph):
        return None
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


def captures_traced_value(fn: Callable, ctx: TraceContext | None) -> bool:
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

"""Helpers for mapping traced references inside literal containers."""

from __future__ import annotations

from typing import Any, Callable

from ..errors import TracingError
from .graph import InputRef, NodeRef, TracedValue
from ..traceable import map_traceable_refs


def normalize_traced_literal(value: Any, inputs: list[TracedValue], ctx: Any) -> Any:
    """Normalize TracedValue objects in literal containers to refs."""
    input_index = {item: idx for idx, item in enumerate(inputs)}

    def resolve_value(item: TracedValue) -> Any:
        if item in input_index:
            return InputRef(input_index[item])
        if item.trace_ctx is ctx:
            return NodeRef(item.node_id)
        raise TracingError("TracedValue from wrong context in literal.")

    return map_traceable_refs(value, (TracedValue,), resolve_value)


def resolve_literal_refs(value: Any, resolve_ref: Callable[[Any], Any]) -> Any:
    """Resolve InputRef/NodeRef objects inside literal containers."""

    return map_traceable_refs(value, (InputRef, NodeRef), resolve_ref)

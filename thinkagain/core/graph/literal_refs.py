"""Helpers for mapping traced references inside literal containers."""

from __future__ import annotations

from typing import Any, Callable

from ..errors import TracingError
from .graph import InputRef, NodeRef, TracedValue
from .literals import map_literal


def normalize_traced_literal(value: Any, inputs: list[TracedValue], ctx: Any) -> Any:
    """Normalize TracedValue objects in literal containers to refs."""

    def mapper(item: Any) -> tuple[bool, Any]:
        if isinstance(item, TracedValue):
            if item in inputs:
                return True, InputRef(inputs.index(item))
            if item.trace_ctx is ctx:
                return True, NodeRef(item.node_id)
            raise TracingError("TracedValue from wrong context in literal.")
        return False, None

    return map_literal(value, mapper)


def resolve_literal_refs(value: Any, resolve_ref: Callable[[Any], Any]) -> Any:
    """Resolve InputRef/NodeRef objects inside literal containers."""

    def mapper(item: Any) -> tuple[bool, Any]:
        if isinstance(item, (InputRef, NodeRef)):
            return True, resolve_ref(item)
        return False, None

    return map_literal(value, mapper)

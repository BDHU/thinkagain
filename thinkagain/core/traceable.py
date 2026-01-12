"""Traceable container support for nested graph values."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable

from .graph.graph import TracedValue


@dataclass(frozen=True)
class TraceSpec:
    cls: type
    decompose: Callable[[Any], tuple[list[Any], Any]]
    compose: Callable[[Any, list[Any]], Any]


_TRACE_REGISTRY: dict[type, TraceSpec] = {}
_TRACE_CACHE: dict[type, TraceSpec | None] = {}


def register_traceable(
    cls: type,
    *,
    decompose: Callable[[Any], tuple[list[Any], Any]],
    compose: Callable[[Any, list[Any]], Any],
) -> None:
    """Register a class as traceable for nested graph capture."""
    if cls in _TRACE_REGISTRY:
        raise ValueError(f"{cls.__name__} is already registered as traceable")
    _TRACE_REGISTRY[cls] = TraceSpec(
        cls=cls,
        decompose=decompose,
        compose=compose,
    )
    _TRACE_CACHE.pop(cls, None)


def trace(cls: type) -> type:
    """Decorator to mark a class as traceable for nested graph capture.

    The class must define how to decompose/compose in one of two ways:

    1. Protocol methods (for custom classes):
        @trace
        class Point:
            def __init__(self, x: float, y: float):
                self.x = x
                self.y = y

            def decompose(self) -> tuple[list[Any], Any]:
                return [self.x, self.y], None

            @classmethod
            def compose(cls, aux: Any, children: list[Any]) -> Self:
                return cls(children[0], children[1])

    2. Auto-detection for dataclasses (zero boilerplate):
        @trace
        @dataclass
        class State:
            count: int
            items: list[str]

    Note: For classes you don't control, use register_traceable() instead.

    Args:
        cls: The class to mark as traceable

    Returns:
        The decorated class, now traceable in @jit functions
    """
    # Priority 1: Check for protocol methods
    has_decompose = hasattr(cls, "decompose")
    has_compose = hasattr(cls, "compose")

    if has_decompose or has_compose:
        if not (has_decompose and has_compose):
            raise TypeError(
                f"{cls.__name__} must define both decompose and "
                f"compose methods, not just one"
            )

        # Wrap the methods to match the registry signature
        decompose_fn = getattr(cls, "decompose")
        compose_fn = getattr(cls, "compose")

        def _decompose(obj: Any) -> tuple[list[Any], Any]:
            return decompose_fn(obj)

        def _compose(aux: Any, children: list[Any]) -> Any:
            return compose_fn(aux, children)

        register_traceable(cls, decompose=_decompose, compose=_compose)
        return cls

    # Priority 2: Auto-detect for dataclasses
    if is_dataclass(cls):
        field_names = tuple(f.name for f in fields(cls) if f.init)

        def _decompose(obj: Any) -> tuple[list[Any], Any]:
            return [getattr(obj, name) for name in field_names], field_names

        def _compose(aux: Any, children: list[Any]) -> Any:
            return cls(**dict(zip(aux, children)))

        register_traceable(cls, decompose=_decompose, compose=_compose)
        return cls

    # No method found
    raise TypeError(
        f"Cannot infer how to trace {cls.__name__}. Either:\n"
        f"  1. Define decompose/compose methods\n"
        f"  2. Make it a @dataclass\n"
        f"  3. Use register_traceable() for classes you don't control"
    )


def _contains_traced_value(value: Any, *, _seen: set[int]) -> bool:
    if isinstance(value, TracedValue):
        return True
    obj_id = id(value)
    if obj_id in _seen:
        return False
    _seen.add(obj_id)

    spec = _get_trace_spec(value)
    if spec is not None:
        children, _aux = spec.decompose(value)
        return any(_contains_traced_value(child, _seen=_seen) for child in children)

    if isinstance(value, dict):
        return any(
            _contains_traced_value(k, _seen=_seen)
            or _contains_traced_value(v, _seen=_seen)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_traced_value(item, _seen=_seen) for item in value)
    return False


def _get_trace_spec(value: Any) -> TraceSpec | None:
    value_type = type(value)
    if value_type in _TRACE_CACHE:
        return _TRACE_CACHE[value_type]

    spec = _TRACE_REGISTRY.get(value_type)
    if spec is not None:
        _TRACE_CACHE[value_type] = spec
        return spec

    _TRACE_CACHE[value_type] = None
    return None


def map_traceable(value: Any, mapper: Callable[[Any], tuple[bool, Any]]) -> Any:
    """Recursively map values across registered traceable containers."""
    handled, new_value = mapper(value)
    if handled:
        return new_value

    if isinstance(value, dict):
        if any(_contains_traced_value(key, _seen=set()) for key in value.keys()):
            raise TypeError(
                "TracedValue is not supported in dict keys. "
                "Use values or a different container."
            )
        return {k: map_traceable(v, mapper) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        if any(_contains_traced_value(item, _seen=set()) for item in value):
            raise TypeError(
                "TracedValue is not supported in sets. Use list or tuple instead."
            )
        return value
    if isinstance(value, list):
        return [map_traceable(v, mapper) for v in value]
    if isinstance(value, tuple):
        return tuple(map_traceable(v, mapper) for v in value)

    spec = _get_trace_spec(value)
    if spec is not None:
        children, aux = spec.decompose(value)
        mapped_children = [map_traceable(child, mapper) for child in children]
        return spec.compose(aux, mapped_children)

    return value


def map_traceable_refs(
    value: Any,
    ref_types: tuple[type, ...],
    resolver: Callable[[Any], Any],
) -> Any:
    """Map traceable containers, resolving specific ref types."""

    def mapper(item: Any) -> tuple[bool, Any]:
        if isinstance(item, ref_types):
            return True, resolver(item)
        return False, None

    return map_traceable(value, mapper)

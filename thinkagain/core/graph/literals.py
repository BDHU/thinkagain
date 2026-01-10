"""Helpers for literal container mapping."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


def map_literal(value: Any, mapper: Callable[[Any], tuple[bool, T]]) -> Any:
    """Recursively map a literal container using a mapper.

    The mapper returns (handled, new_value). If handled is True, new_value is
    returned directly. Otherwise, containers are traversed and mapped.
    """
    handled, new_value = mapper(value)
    if handled:
        return new_value

    if isinstance(value, dict):
        return {k: map_literal(v, mapper) for k, v in value.items()}
    if isinstance(value, list):
        return [map_literal(v, mapper) for v in value]
    if isinstance(value, tuple):
        return tuple(map_literal(v, mapper) for v in value)
    return value

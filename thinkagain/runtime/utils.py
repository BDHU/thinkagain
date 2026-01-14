"""Runtime helpers shared by tracing and execution."""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable

from .object_ref import ObjectRef


async def maybe_await(fn: Callable, *args, **kwargs) -> Any:
    """Call a function and await if it returns an awaitable."""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def extract_object_refs(value: Any) -> list[ObjectRef]:
    """Recursively extract ObjectRefs from nested containers."""
    refs: list[ObjectRef] = []

    if isinstance(value, ObjectRef):
        refs.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            refs.extend(extract_object_refs(item))
    elif isinstance(value, dict):
        for item in value.values():
            refs.extend(extract_object_refs(item))

    return refs


async def resolve_object_refs(
    value: Any,
    resolver: Callable[[ObjectRef], Awaitable[Any]] | None = None,
) -> Any:
    """Recursively resolve ObjectRefs in nested containers."""
    if resolver is None:

        def resolver(ref):
            return ref.get()

    if isinstance(value, ObjectRef):
        return await resolver(value)
    if isinstance(value, list):
        return [await resolve_object_refs(item, resolver) for item in value]
    if isinstance(value, tuple):
        items = [await resolve_object_refs(item, resolver) for item in value]
        return tuple(items)
    if isinstance(value, dict):
        result = {}
        for key, val in value.items():
            result[key] = await resolve_object_refs(val, resolver)
        return result
    return value

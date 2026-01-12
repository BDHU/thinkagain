"""Replica state update helpers for decompose/compose workflows."""

from __future__ import annotations

from typing import Any, Iterable

from .runtime import maybe_await


def _iter_slots(cls: type) -> Iterable[str]:
    for base in cls.__mro__:
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot not in ("__dict__", "__weakref__"):
                yield slot


def _copy_replica_state(target: Any, source: Any) -> None:
    if type(target) is not type(source):
        raise TypeError(
            "Replica compose() must return the same class as the target replica."
        )

    if hasattr(target, "__dict__"):
        target.__dict__.clear()
        target.__dict__.update(source.__dict__)

    for slot in _iter_slots(type(target)):
        if hasattr(source, slot):
            setattr(target, slot, getattr(source, slot))


async def apply_replica(replica_obj: Any, fn: Any, *args, **kwargs) -> Any:
    """Apply a @jit-compatible state update to a replica via decompose/compose.

    The replica must implement:
      - decompose(self) -> (children: list[Any] | tuple[Any, ...], aux: Any)
      - compose(cls, aux, children) -> replica instance

    The function must return (new_children, output), where new_children matches
    the structure of decompose()'s children.
    """
    if not hasattr(replica_obj, "decompose") or not hasattr(
        replica_obj.__class__, "compose"
    ):
        raise TypeError(
            "Replica must define decompose() and compose() to use apply_replica()."
        )

    children, aux = replica_obj.decompose()
    if not isinstance(children, (list, tuple)):
        raise TypeError("decompose() must return a list/tuple of children.")

    result = await maybe_await(fn, *children, *args, **kwargs)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise TypeError("Expected (new_children, output) from replica update.")

    new_children, output = result
    if not isinstance(new_children, (list, tuple)):
        new_children = [new_children]
    if len(new_children) != len(children):
        raise ValueError(
            "Updated children count must match decompose() children count."
        )

    updated = type(replica_obj).compose(aux, list(new_children))
    _copy_replica_state(replica_obj, updated)
    return output

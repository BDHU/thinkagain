"""Serialization interface for distributed backends."""

from __future__ import annotations

import pickle
from typing import Protocol


class Serializer(Protocol):
    """Protocol for serializer implementations."""

    def dumps(self, obj: object) -> bytes: ...
    def loads(self, payload: bytes) -> object: ...


class PickleSerializer:
    """Default serializer using pickle (supports arbitrary Python objects)."""

    def dumps(self, obj: object) -> bytes:
        return pickle.dumps(obj)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)

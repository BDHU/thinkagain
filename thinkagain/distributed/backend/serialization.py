"""Serialization interface for distributed backends."""

from __future__ import annotations

import pickle
from typing import Protocol


class Serializer(Protocol):
    """Protocol for serializer implementations."""

    def dumps(self, obj: object) -> bytes: ...
    def loads(self, payload: bytes) -> object: ...


class PickleSerializer:
    """Default serializer using pickle (supports arbitrary Python objects).

    SECURITY WARNING

    Pickle can execute arbitrary code during deserialization. Do NOT use
    this serializer with untrusted gRPC clients or over untrusted networks.

    For production use with untrusted sources, implement a custom Serializer
    using a safer format such as:
    - JSON (for simple data types)
    - Protocol Buffers (for structured data)
    - MessagePack (for binary efficiency)

    Example custom serializer:
        class JsonSerializer:
            def dumps(self, obj: object) -> bytes:
                return json.dumps(obj).encode('utf-8')

            def loads(self, payload: bytes) -> object:
                return json.loads(payload.decode('utf-8'))

    See: https://docs.python.org/3/library/pickle.html#module-pickle
    """

    def dumps(self, obj: object) -> bytes:
        return pickle.dumps(obj)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)

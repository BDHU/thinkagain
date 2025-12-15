"""Minimal context - just a dict wrapper with attribute access."""

from typing import Any


class Context(dict):
    """Dict with attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

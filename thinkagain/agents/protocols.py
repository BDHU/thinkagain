"""LLM provider interface types."""

from __future__ import annotations

from typing import Any, Protocol


class LLM(Protocol):
    """Interface for chat-style models."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a message dict with optional tool calls."""
        ...

"""Core agent state types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    """A tool invocation request.

    Args:
        id: Unique identifier for this tool call
        name: Name of the tool to invoke
        arguments: Dictionary of arguments to pass to the tool
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A single chat message.

    Args:
        role: The role of the message sender (user, assistant, tool, or system)
        content: The message content (text or multimodal content list)
        tool_calls: Optional list of tool calls to execute
        tool_call_id: Optional ID linking this message to a tool call (for tool role)
    """

    role: Literal["user", "assistant", "tool", "system"]
    content: str | list[dict[str, Any]]
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class AgentState:
    """Agent state container.

    Args:
        messages: Conversation history
        metadata: Optional metadata for tracking costs, timing, custom data
    """

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

"""LLM provider helpers and core agent nodes."""

from __future__ import annotations

import traceback
from typing import Any, Callable

from thinkagain.core.node import node

from .protocols import LLM
from .state import AgentState, Message, ToolCall
from .tools import _function_to_schema


def _message_to_dict(msg: Message) -> dict[str, Any]:
    """Convert internal message to provider dict."""
    data: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        data["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id:
        data["tool_call_id"] = msg.tool_call_id
    return data


def _parse_tool_calls(response: dict[str, Any]) -> list[ToolCall] | None:
    """Parse provider tool calls into ToolCall objects."""
    tool_calls = response.get("tool_calls") or []
    if not tool_calls:
        return None
    return [
        ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
        for tc in tool_calls
    ]


def _build_tool_map(tools: list[Callable] | None) -> dict[str, Callable]:
    """Build a name -> function mapping from tool list."""
    if not tools:
        return {}

    tool_map = {}
    for fn in tools:
        name = getattr(fn, "__tool_name__", fn.__name__)
        if name in tool_map:
            raise ValueError(f"Duplicate tool name: {name}")
        tool_map[name] = fn
    return tool_map


@node
async def call_llm(
    state: AgentState,
    provider: LLM,
    tools: list[Callable] | None = None,
    **kwargs: Any,
) -> AgentState:
    """Call LLM and return updated state with assistant message.

    Args:
        state: Current agent state
        provider: LLM provider instance
        tools: Optional list of tool functions
        **kwargs: Additional arguments for LLM provider

    Returns:
        Updated agent state with new assistant message
    """
    messages_dict = [_message_to_dict(msg) for msg in state.messages]

    tool_map = _build_tool_map(tools)
    tool_schemas = (
        [_function_to_schema(fn) for fn in tool_map.values()] if tool_map else None
    )

    response = await provider.complete(messages_dict, tools=tool_schemas, **kwargs)

    tool_calls = _parse_tool_calls(response)

    message = Message(
        role=response["role"],
        content=response["content"],
        tool_calls=tool_calls,
    )

    return AgentState(messages=state.messages + [message], metadata=state.metadata)


@node
async def execute_tool_call(
    state: AgentState,
    tool_call: ToolCall,
    tool_fn: Callable,
) -> AgentState:
    """Execute a single tool call and append result to state.

    This is a granular @node that can be called independently or in parallel.

    Args:
        state: Current agent state
        tool_call: The tool call to execute
        tool_fn: The tool function to invoke

    Returns:
        Updated agent state with tool result message appended
    """
    try:
        result = await tool_fn(**tool_call.arguments)
        result_content = str(result)
    except Exception as e:
        result_content = (
            f"Error executing tool '{tool_call.name}': {str(e)}\n"
            f"{traceback.format_exc()}"
        )

    result_message = Message(
        role="tool",
        content=result_content,
        tool_call_id=tool_call.id,
    )

    return AgentState(
        messages=state.messages + [result_message],
        metadata=state.metadata,
    )


async def execute_tools(
    state: AgentState,
    tools: list[Callable],
) -> AgentState:
    """Execute all tool calls from last message.

    This orchestrates multiple execute_tool_call @nodes, allowing them
    to run in parallel naturally through the ThinkAgain framework.

    Args:
        state: Current agent state
        tools: List of tool functions

    Returns:
        Updated agent state with all tool results
    """
    from thinkagain import Context

    last_message = state.messages[-1]

    if not last_message.tool_calls:
        return state

    tool_map = _build_tool_map(tools)

    # Create @node contexts for each tool call
    contexts = []
    for tool_call in last_message.tool_calls:
        if tool_call.name not in tool_map:
            # Handle missing tool inline (no need for @node)
            error_msg = Message(
                role="tool",
                content=f"Error: Tool '{tool_call.name}' not found. "
                f"Available tools: {list(tool_map.keys())}",
                tool_call_id=tool_call.id,
            )
            state = AgentState(
                messages=state.messages + [error_msg],
                metadata=state.metadata,
            )
        else:
            # Create a context for this tool execution
            tool_fn = tool_map[tool_call.name]
            ctx = execute_tool_call(
                Context(state), tool_call=tool_call, tool_fn=tool_fn
            )
            contexts.append(ctx)

    # Execute all tool calls (ThinkAgain handles parallelism)
    for ctx in contexts:
        state = await ctx

    return state

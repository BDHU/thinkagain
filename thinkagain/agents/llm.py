"""LLM provider interface and core agent nodes."""

from __future__ import annotations

import inspect
import traceback
from typing import Any, Callable, Protocol

from thinkagain.core.node import node

from .state import AgentState, Message, ToolCall


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


_SIMPLE_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def tool(
    _fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable:
    """Decorator to mark a function as a tool and optionally name it."""

    def wrapper(fn: Callable) -> Callable:
        tool_name = name or getattr(fn, "__name__", "tool")
        setattr(fn, "__tool_name__", tool_name)
        if description is not None:
            setattr(fn, "__tool_description__", description)
        return fn

    if _fn is None:
        return wrapper

    return wrapper(_fn)


def _get_type_name(annotation: Any) -> dict[str, Any]:
    """Convert Python type annotation to JSON schema type."""
    from typing import get_args, get_origin

    if annotation is type(None):
        return {"type": "null"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional[T] and Union types
    if origin is type(None) or (origin and str(origin) == "typing.Union"):
        if (
            args
            and (non_none := [a for a in args if a is not type(None)])
            and len(non_none) == 1
        ):
            return _get_type_name(non_none[0])

    # Handle list[T]
    if origin is list or annotation is list:
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _get_type_name(args[0])
        return schema

    # Handle dict
    if origin is dict or annotation is dict:
        return {"type": "object"}

    # Handle simple types
    return {"type": _SIMPLE_TYPE_MAP.get(annotation, "string")}


def _parse_google_docstring(doc: str) -> dict[str, str]:
    """Parse Google-style docstring to extract parameter descriptions."""
    if not doc:
        return {}

    param_descriptions = {}
    in_args_section = False
    current_param = None

    for line in doc.split("\n"):
        stripped = line.strip()

        # Check section transitions
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args_section = True
            continue
        if (
            in_args_section
            and stripped
            and stripped.endswith(":")
            and not line.startswith(" ")
        ):
            break

        # Parse parameter lines
        if in_args_section and stripped:
            if ":" in stripped:
                param_part, desc_part = stripped.split(":", 1)
                param_name = param_part.split("(")[0].strip()
                if param_name:
                    current_param = param_name
                    param_descriptions[param_name] = desc_part.strip()
            elif current_param:
                param_descriptions[current_param] += " " + stripped

    return param_descriptions


def _function_to_schema(fn: Callable) -> dict[str, Any]:
    """Generate a tool schema from a function signature."""
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    param_descriptions = _parse_google_docstring(doc)

    # Extract description from docstring or use custom description
    description_lines = [
        line.strip()
        for line in doc.split("\n")
        if (stripped := line.strip()) and not stripped.endswith(":")
    ]
    if description_lines and description_lines[0]:
        # Stop at first empty line or section marker
        first_para = []
        for line in description_lines:
            if not line:
                break
            first_para.append(line)
        description = " ".join(first_para) if first_para else fn.__name__
    else:
        description = fn.__name__
    description = getattr(fn, "__tool_description__", description)

    # Build parameters schema
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        type_schema = (
            _get_type_name(param.annotation)
            if param.annotation != inspect.Parameter.empty
            else {"type": "string"}
        )

        if param_name in param_descriptions:
            type_schema["description"] = param_descriptions[param_name]

        properties[param_name] = type_schema

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": getattr(fn, "__tool_name__", fn.__name__),
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


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

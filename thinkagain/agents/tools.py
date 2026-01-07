"""Tool decorator and schema helpers for LLMs."""

from __future__ import annotations

import inspect
from typing import Any, Callable


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


_SIMPLE_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


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

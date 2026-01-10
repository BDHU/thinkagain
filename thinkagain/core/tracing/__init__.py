"""JAX-style tracing infrastructure for graph capture.

This module provides the tracing system that captures computation graphs
when functions decorated with @jit are called.
"""

from .context import TraceContext, get_trace_context, is_tracing
from .tracer import (
    clear_compiled_cache,
    get_cache_info,
    jit,
    node,
    register_tracing_plugin,
    trace_branch,
)
from .utils import captures_traced_value, get_source_location, is_traceable
from .validation import (
    validate_cond_branches,
    validate_scan_body,
    validate_switch_branches,
    validate_while_body,
)

__all__ = [
    # Main decorators
    "jit",
    "node",
    # Context management
    "TraceContext",
    "get_trace_context",
    "is_tracing",
    # Utilities
    "clear_compiled_cache",
    "get_cache_info",
    "get_source_location",
    "is_traceable",
    "captures_traced_value",
    "register_tracing_plugin",
    "trace_branch",
    # Validation
    "validate_cond_branches",
    "validate_scan_body",
    "validate_switch_branches",
    "validate_while_body",
]

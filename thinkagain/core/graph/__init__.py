"""Graph data structures and operations.

This module provides the core graph representation and control flow operations
for computation graphs.
"""

from .graph import (
    Graph,
    InputRef,
    Node,
    NodeRef,
    OutputKind,
    OutputRef,
    TracedValue,
)
from .literal_refs import normalize_traced_literal
from .ops import cond, scan, switch, while_loop

__all__ = [
    # Graph types
    "Graph",
    "Node",
    "NodeRef",
    "InputRef",
    "OutputRef",
    "OutputKind",
    "TracedValue",
    # Literals
    "normalize_traced_literal",
    # Control flow operations
    "cond",
    "while_loop",
    "scan",
    "switch",
]

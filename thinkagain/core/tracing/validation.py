"""Validation helpers for traced graphs and control flow."""

from __future__ import annotations

from typing import Callable

from ..errors import TracingError
from ..graph.graph import Graph, OutputKind


def _validate_graph(graph: Graph | Callable, context: str) -> None:
    """Validate a graph or callable for control flow."""
    if not isinstance(graph, Graph):
        return
    if graph.output_ref is None:
        raise TracingError(f"{context} must return a value.")
    if graph.output_ref.kind is OutputKind.LITERAL and graph.output_ref.value is None:
        raise TracingError(f"{context} returns None.")


def validate_cond_branches(
    true_branch: Graph | Callable, false_branch: Graph | Callable
) -> None:
    """Validate cond branches have compatible outputs."""
    _validate_graph(true_branch, "cond true branch")
    _validate_graph(false_branch, "cond false branch")
    if isinstance(true_branch, Graph) and isinstance(false_branch, Graph):
        if true_branch.output_ref.kind is not false_branch.output_ref.kind:
            raise TracingError(
                f"cond branches must return same pattern: "
                f"true={true_branch.output_ref.kind.value}, "
                f"false={false_branch.output_ref.kind.value}"
            )


def validate_while_body(body: Graph | Callable) -> None:
    """Validate while_loop body."""
    _validate_graph(body, "while_loop body")


def validate_scan_body(body: Graph | Callable) -> None:
    """Validate scan body."""
    _validate_graph(body, "scan body")
    if isinstance(body, Graph) and body.output_ref.kind is OutputKind.LITERAL:
        val = body.output_ref.value
        if not isinstance(val, tuple) or len(val) != 2:
            raise TracingError(f"scan body must return (carry, output), got {val!r}")


def validate_switch_branches(branches: list[Graph | Callable]) -> None:
    """Validate switch branches have compatible outputs."""
    if not branches:
        raise TracingError("switch must have at least one branch")

    for i, branch in enumerate(branches):
        _validate_graph(branch, f"switch branch {i}")

    graph_branches = [b for b in branches if isinstance(b, Graph)]
    if len(graph_branches) > 1:
        first_kind = graph_branches[0].output_ref.kind
        for i, branch in enumerate(graph_branches[1:], start=1):
            if branch.output_ref.kind is not first_kind:
                raise TracingError(
                    f"switch branches must return same pattern: "
                    f"branch 0={first_kind.value}, "
                    f"branch {i}={branch.output_ref.kind.value}"
                )

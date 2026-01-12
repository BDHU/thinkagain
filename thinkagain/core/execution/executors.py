"""Node executor implementations.

Each executor is a frozen dataclass that implements the NodeExecutor protocol.
Executors encapsulate both the execution logic and the data needed for that
execution (e.g., the function to call, branches to take, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from .runtime import maybe_await

if TYPE_CHECKING:
    from .executor import ExecutionContext


async def _execute_branch(
    branch: Any, ctx: ExecutionContext, operand_args: tuple
) -> Any:
    from ..graph.graph import Graph

    if isinstance(branch, Graph):
        graph_args = ctx.prepare_subgraph_args(branch, operand_args)
        return await branch.execute(
            *graph_args,
            parent_values=ctx.capture_values,
            service_provider=ctx.service_provider,
        )
    return await maybe_await(branch, *operand_args)


def _expect_pair(result: Any, *, context: str) -> tuple[Any, Any]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"{context} must return (carry, output), got {result!r}")
    return result


def _select_branch(index: Any, branches: tuple) -> Any:
    if not isinstance(index, int) or index < 0 or index >= len(branches):
        raise IndexError(f"switch index {index} out of range [0, {len(branches)})")
    return branches[index]


# ---------------------------------------------------------------------------
# Call Executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallExecutor:
    """Executor for regular function calls."""

    fn: Callable

    async def execute(self, args: tuple, kwargs: dict, ctx: ExecutionContext) -> Any:
        from .hooks import _hooks

        # Check hooks for custom execution (e.g., distributed)
        if _hooks:
            for hook in _hooks:
                handled, result = await hook(self.fn, args, kwargs, id(self))
                if handled:
                    return result

        # Execute the function directly
        # Service bindings (if any) are captured via normal Python closures
        return await maybe_await(self.fn, *args, **kwargs)

    def display_name(self) -> str:
        return getattr(self.fn, "__name__", type(self.fn).__name__)


# ---------------------------------------------------------------------------
# Cond Executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CondExecutor:
    """Executor for conditional branching."""

    pred_fn: Callable
    true_branch: Any  # Graph | Callable
    false_branch: Any  # Graph | Callable

    async def execute(self, args: tuple, kwargs: dict, ctx: ExecutionContext) -> Any:
        operand = args[0]
        pred_result = await maybe_await(self.pred_fn, operand)
        branch = self.true_branch if pred_result else self.false_branch

        return await _execute_branch(branch, ctx, (operand,))

    def display_name(self) -> str:
        return "cond"


# ---------------------------------------------------------------------------
# While Executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhileExecutor:
    """Executor for while loops."""

    cond_fn: Callable
    body_fn: Any  # Graph | Callable

    async def execute(self, args: tuple, kwargs: dict, ctx: ExecutionContext) -> Any:
        operand = args[0]

        while await maybe_await(self.cond_fn, operand):
            operand = await _execute_branch(self.body_fn, ctx, (operand,))
        return operand

    def display_name(self) -> str:
        return "while"


# ---------------------------------------------------------------------------
# Scan Executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanExecutor:
    """Executor for scan (fold with outputs) operations."""

    body_fn: Any  # Graph | Callable

    async def execute(self, args: tuple, kwargs: dict, ctx: ExecutionContext) -> Any:
        carry, xs = args[0], args[1]
        outputs = []

        for x in xs:
            result = await _execute_branch(self.body_fn, ctx, (carry, x))
            carry, output = _expect_pair(result, context="scan body")
            outputs.append(output)

        return (carry, outputs)

    def display_name(self) -> str:
        return "scan"


# ---------------------------------------------------------------------------
# Switch Executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwitchExecutor:
    """Executor for multi-way branching."""

    index_fn: Callable
    branches: tuple  # tuple[Graph | Callable, ...] - tuple for frozen dataclass

    async def execute(self, args: tuple, kwargs: dict, ctx: ExecutionContext) -> Any:
        operand = args[0]
        index = await maybe_await(self.index_fn, operand)
        branch = _select_branch(index, self.branches)
        return await _execute_branch(branch, ctx, (operand,))

    def display_name(self) -> str:
        return "switch"

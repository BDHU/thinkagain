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
        from .graph import Graph

        operand = args[0]
        pred_result = await maybe_await(self.pred_fn, operand)
        branch = self.true_branch if pred_result else self.false_branch

        if isinstance(branch, Graph):
            graph_args = ctx.prepare_subgraph_args(branch, (operand,))
            return await branch.execute(
                *graph_args,
                parent_values=ctx.capture_values,
                service_provider=ctx.service_provider,
            )
        return await maybe_await(branch, operand)

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
        from .graph import Graph

        operand = args[0]

        while await maybe_await(self.cond_fn, operand):
            if isinstance(self.body_fn, Graph):
                graph_args = ctx.prepare_subgraph_args(self.body_fn, (operand,))
                operand = await self.body_fn.execute(
                    *graph_args,
                    parent_values=ctx.capture_values,
                    service_provider=ctx.service_provider,
                )
            else:
                operand = await maybe_await(self.body_fn, operand)
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
        from .graph import Graph

        carry, xs = args[0], args[1]
        outputs = []

        for x in xs:
            if isinstance(self.body_fn, Graph):
                graph_args = ctx.prepare_subgraph_args(self.body_fn, (carry, x))
                result = await self.body_fn.execute(
                    *graph_args,
                    parent_values=ctx.capture_values,
                    service_provider=ctx.service_provider,
                )
            else:
                result = await maybe_await(self.body_fn, carry, x)

            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError(
                    f"scan body must return (carry, output) tuple, got {result!r}"
                )
            carry, output = result
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
        from .graph import Graph

        operand = args[0]
        index = await maybe_await(self.index_fn, operand)

        if not isinstance(index, int) or index < 0 or index >= len(self.branches):
            raise IndexError(
                f"switch index {index} out of range [0, {len(self.branches)})"
            )

        branch = self.branches[index]

        if isinstance(branch, Graph):
            graph_args = ctx.prepare_subgraph_args(branch, (operand,))
            return await branch.execute(
                *graph_args,
                parent_values=ctx.capture_values,
                service_provider=ctx.service_provider,
            )
        return await maybe_await(branch, operand)

    def display_name(self) -> str:
        return "switch"

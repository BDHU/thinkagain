"""
Pipeline and control flow components for building computation graphs.

Provides Pipeline, Conditional, and Loop for composing workers into
complex workflows with explicit control flow.

Each component supports both synchronous and asynchronous execution:
- Use .run(ctx) for synchronous execution
- Use await .arun(ctx) for asynchronous execution
"""

import asyncio
from typing import Callable, List, Union, Tuple, Optional, Any
from .context import Context


class Pipeline:
    """
    A pipeline is a sequence of workers/nodes that execute in order.

    Pipelines can be composed with workers using >> operator.
    The pipeline maintains the computation graph structure for inspection.

    Supports both sync and async execution modes:

    Synchronous example:
        pipeline = worker1 >> worker2 >> worker3
        ctx = pipeline.run(Context(query="test"))

    Asynchronous example:
        pipeline = async_worker1 >> async_worker2 >> async_worker3
        ctx = await pipeline.arun(Context(query="test"))
    """

    def __init__(self, nodes: List = None, name: str = "pipeline"):
        """
        Initialize pipeline with a list of nodes.

        Args:
            nodes: List of workers/conditionals/loops
            name: Name for this pipeline
        """
        self.name = name
        self.nodes = nodes or []

    def __call__(self, ctx: Context) -> Context:
        """Execute all nodes in sequence (synchronous). Alias for run()."""
        return self.run(ctx)

    def run(self, ctx: Context) -> Context:
        """Execute all nodes synchronously."""
        ctx.log(f"[Pipeline] Starting: {self.name}")

        for node in self.nodes:
            ctx = node(ctx)

        ctx.log(f"[Pipeline] Completed: {self.name}")
        return ctx

    async def arun(self, ctx: Context) -> Context:
        """Execute all nodes asynchronously."""
        ctx.log(f"[Pipeline] Starting: {self.name}")

        for node in self.nodes:
            # Call arun() for async control flow (Pipeline/Switch/Loop)
            # Call acall() for async workers
            # Otherwise await __call__ for async functions
            if hasattr(node, 'arun'):
                ctx = await node.arun(ctx)
            elif hasattr(node, 'acall'):
                ctx = await node.acall(ctx)
            else:
                ctx = await node(ctx)

        ctx.log(f"[Pipeline] Completed: {self.name}")
        return ctx

    def __rshift__(self, other) -> 'Pipeline':
        """Extend pipeline with another worker/pipeline."""
        if isinstance(other, Pipeline):
            return Pipeline(
                nodes=self.nodes + other.nodes,
                name=f"{self.name}_extended"
            )
        return Pipeline(
            nodes=self.nodes + [other],
            name=f"{self.name}_extended"
        )

    def to_dict(self) -> dict:
        """Export full pipeline structure as dictionary."""
        return {
            "type": "Pipeline",
            "name": self.name,
            "nodes": [
                node.to_dict() if hasattr(node, 'to_dict')
                else {"type": "Unknown", "name": str(node)}
                for node in self.nodes
            ]
        }

    def visualize(self, indent: int = 0) -> str:
        """
        Generate ASCII visualization of the pipeline structure.

        Returns:
            String representation of pipeline graph
        """
        lines = []
        prefix = "  " * indent

        lines.append(f"{prefix}Pipeline: {self.name}")

        for i, node in enumerate(self.nodes):
            is_last = i == len(self.nodes) - 1
            connector = "└──" if is_last else "├──"

            if isinstance(node, Pipeline):
                lines.append(f"{prefix}{connector} (sub-pipeline)")
                lines.append(node.visualize(indent + 1))
            elif isinstance(node, Switch):
                lines.append(f"{prefix}{connector} Switch: {node.name}")
                for j, (cond_name, branch) in enumerate(node.cases):
                    is_last_case = j == len(node.cases) - 1 and node.default is None
                    case_connector = "└──" if is_last_case else "├──"
                    branch_name = branch.name if hasattr(branch, 'name') else 'branch'
                    lines.append(f"{prefix}    {case_connector} CASE {j+1}: {branch_name}")
                if node.default:
                    default_name = node.default.name if hasattr(node.default, 'name') else 'branch'
                    lines.append(f"{prefix}    └── DEFAULT: {default_name}")
            elif isinstance(node, Loop):
                lines.append(f"{prefix}{connector} Loop: {node.name} (max={node.max_iterations})")
                body_name = node.body.name if hasattr(node.body, 'name') else 'body'
                lines.append(f"{prefix}    └── Body: {body_name}")
            elif hasattr(node, 'workers'):  # Parallel node
                lines.append(f"{prefix}{connector} Parallel: {node.name}")
                for j, worker in enumerate(node.workers):
                    is_last_worker = j == len(node.workers) - 1
                    worker_connector = "└──" if is_last_worker else "├──"
                    worker_name = worker.name if hasattr(worker, 'name') else str(worker)
                    lines.append(f"{prefix}    {worker_connector} {worker_name}")
            else:
                node_name = node.name if hasattr(node, 'name') else str(node)
                lines.append(f"{prefix}{connector} {node_name}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Pipeline(name='{self.name}', nodes={len(self.nodes)})"


class Switch:
    """
    Multi-way conditional branching (if/elif/elif/.../else).

    Evaluates conditions in order and executes the first matching branch.
    If no condition matches, executes the default branch (if provided).

    Example 1: Simple if/else (binary)
        pipeline = (
            retrieve
            >> Switch(
                cases=[
                    (lambda ctx: len(ctx.documents) >= 5, high_quality_path),
                    (lambda ctx: len(ctx.documents) >= 2, medium_quality_path),
                ],
                default=fallback_path
            )
            >> generate
        )

    Example 2: Using helper for readability
        pipeline = (
            retrieve
            >> Switch(name="quality_check")
                .case(lambda ctx: len(ctx.documents) >= 5, high_quality_path)
                .case(lambda ctx: len(ctx.documents) >= 2, medium_quality_path)
                .default(fallback_path)
            >> generate
        )
    """

    def __init__(
        self,
        cases: List[Tuple[Callable[[Context], bool], Any]] = None,
        default: Any = None,
        name: str = "switch"
    ):
        """
        Initialize switch node.

        Args:
            cases: List of (condition_fn, branch) tuples
            default: Default branch if no condition matches
            name: Name for this switch
        """
        self.name = name
        self.cases = cases or []
        self.default = default

    def case(self, condition: Callable[[Context], bool], branch) -> 'Switch':
        """
        Add a case to the switch (builder pattern).

        Args:
            condition: Function that takes Context and returns bool
            branch: Worker/Pipeline to execute if condition is True

        Returns:
            Self for chaining
        """
        self.cases.append((condition, branch))
        return self

    def set_default(self, branch) -> 'Switch':
        """
        Set default branch (builder pattern).

        Args:
            branch: Worker/Pipeline to execute if no condition matches

        Returns:
            Self for chaining
        """
        self.default = branch
        return self

    def __call__(self, ctx: Context) -> Context:
        """Evaluate conditions and execute first matching branch (synchronous). Alias for run()."""
        return self.run(ctx)

    def run(self, ctx: Context) -> Context:
        """Evaluate conditions in order and execute first matching branch (synchronous)."""
        ctx.log(f"[Switch] Evaluating: {self.name}")

        for i, (condition, branch) in enumerate(self.cases):
            result = condition(ctx)
            ctx.log(f"[Switch] Case {i+1} evaluated to: {result}")

            if result:
                ctx.log(f"[Switch] Executing case {i+1}")
                # Call run() if available (for Pipeline/Switch/Loop), else call directly
                if hasattr(branch, 'run'):
                    return branch.run(ctx)
                else:
                    return branch(ctx)

        # No condition matched, execute default if available
        if self.default:
            ctx.log(f"[Switch] No case matched, executing default")
            # Call run() if available (for Pipeline/Switch/Loop), else call directly
            if hasattr(self.default, 'run'):
                return self.default.run(ctx)
            else:
                return self.default(ctx)
        else:
            ctx.log(f"[Switch] No case matched, no default, passing through")
            return ctx

    async def arun(self, ctx: Context) -> Context:
        """Evaluate conditions and execute first matching branch (asynchronously)."""
        ctx.log(f"[Switch] Evaluating: {self.name}")

        for i, (condition, branch) in enumerate(self.cases):
            # Allow async conditions
            if asyncio.iscoroutinefunction(condition):
                result = await condition(ctx)
            else:
                result = condition(ctx)

            ctx.log(f"[Switch] Case {i+1} evaluated to: {result}")

            if result:
                ctx.log(f"[Switch] Executing case {i+1}")
                # Call arun() for async control flow (Pipeline/Switch/Loop)
                # Call acall() for async workers
                # Otherwise await __call__ for async functions
                if hasattr(branch, 'arun'):
                    return await branch.arun(ctx)
                elif hasattr(branch, 'acall'):
                    return await branch.acall(ctx)
                else:
                    return await branch(ctx)

        # Default branch
        if self.default:
            ctx.log(f"[Switch] Executing default")
            # Call arun() for async control flow (Pipeline/Switch/Loop)
            # Call acall() for async workers
            # Otherwise await __call__ for async functions
            if hasattr(self.default, 'arun'):
                return await self.default.arun(ctx)
            elif hasattr(self.default, 'acall'):
                return await self.default.acall(ctx)
            else:
                return await self.default(ctx)

        ctx.log(f"[Switch] No match, passing through")
        return ctx

    def __rshift__(self, other):
        """Allow switches to be composed in pipelines."""
        return Pipeline(nodes=[self, other])

    def to_dict(self) -> dict:
        """Export switch structure."""
        return {
            "type": "Switch",
            "name": self.name,
            "cases": [
                {
                    "condition": cond.__name__ if hasattr(cond, '__name__') else "lambda",
                    "branch": (
                        branch.to_dict() if hasattr(branch, 'to_dict')
                        else {"type": "Unknown", "name": str(branch)}
                    )
                }
                for cond, branch in self.cases
            ],
            "default": (
                self.default.to_dict() if self.default and hasattr(self.default, 'to_dict')
                else {"type": "Unknown", "name": str(self.default)} if self.default
                else None
            )
        }

    def __repr__(self) -> str:
        return f"Switch(name='{self.name}', cases={len(self.cases)}, has_default={self.default is not None})"


# Convenience alias: Conditional = binary Switch
class Conditional(Switch):
    """
    Binary conditional (if/else) - simplified interface for Switch.

    This is just a convenience wrapper around Switch for the common
    two-branch case.

    Example:
        pipeline = (
            retrieve
            >> Conditional(
                condition=lambda ctx: len(ctx.documents) >= 2,
                true_branch=rerank,
                false_branch=fallback
            )
            >> generate
        )
    """

    def __init__(
        self,
        condition: Callable[[Context], bool],
        true_branch,
        false_branch,
        name: str = "conditional"
    ):
        """
        Initialize binary conditional.

        Args:
            condition: Function that takes Context and returns bool
            true_branch: Worker/Pipeline to execute if condition is True
            false_branch: Worker/Pipeline to execute if condition is False
            name: Name for this conditional
        """
        super().__init__(
            cases=[(condition, true_branch)],
            default=false_branch,
            name=name
        )
        # Keep references for easier inspection
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch


class Loop:
    """
    Loop node that repeatedly executes a body while a condition is true.

    Includes max_iterations safeguard to prevent infinite loops.

    Example:
        pipeline = (
            retrieve
            >> Loop(
                condition=lambda ctx: len(ctx.documents) < 2,
                body=refine_query >> retrieve,
                max_iterations=3
            )
            >> generate
        )
    """

    def __init__(
        self,
        condition: Callable[[Context], bool],
        body,
        max_iterations: int = 10,
        name: str = "loop"
    ):
        """
        Initialize loop node.

        Args:
            condition: Function that takes Context and returns bool
            body: Worker/Pipeline to execute repeatedly
            max_iterations: Maximum number of iterations (safety limit)
            name: Name for this loop
        """
        self.name = name
        self.condition = condition
        self.body = body
        self.max_iterations = max_iterations

    def __call__(self, ctx: Context) -> Context:
        """Execute body repeatedly while condition is true (synchronous). Alias for run()."""
        return self.run(ctx)

    def run(self, ctx: Context) -> Context:
        """Execute body repeatedly while condition is true (synchronous)."""
        iteration = 0

        ctx.log(f"[Loop] Starting: {self.name}")

        while iteration < self.max_iterations and self.condition(ctx):
            ctx.log(f"[Loop] Iteration {iteration + 1}/{self.max_iterations}")
            # Call run() if available (for Pipeline/Switch/Loop), else call directly
            if hasattr(self.body, 'run'):
                ctx = self.body.run(ctx)
            else:
                ctx = self.body(ctx)
            iteration += 1

        if iteration >= self.max_iterations:
            ctx.log(f"[Loop] Terminated: max iterations reached")
        else:
            ctx.log(f"[Loop] Completed after {iteration} iterations")

        return ctx

    async def arun(self, ctx: Context) -> Context:
        """Execute body repeatedly while condition is true (asynchronous)."""
        iteration = 0
        ctx.log(f"[Loop] Starting: {self.name}")

        while iteration < self.max_iterations:
            # Allow async conditions
            if asyncio.iscoroutinefunction(self.condition):
                should_continue = await self.condition(ctx)
            else:
                should_continue = self.condition(ctx)

            if not should_continue:
                break

            ctx.log(f"[Loop] Iteration {iteration + 1}/{self.max_iterations}")
            # Call arun() for async control flow (Pipeline/Switch/Loop)
            # Call acall() for async workers
            # Otherwise await __call__ for async functions
            if hasattr(self.body, 'arun'):
                ctx = await self.body.arun(ctx)
            elif hasattr(self.body, 'acall'):
                ctx = await self.body.acall(ctx)
            else:
                ctx = await self.body(ctx)
            iteration += 1

        if iteration >= self.max_iterations:
            ctx.log(f"[Loop] Terminated: max iterations reached")
        else:
            ctx.log(f"[Loop] Completed after {iteration} iterations")

        return ctx

    def __rshift__(self, other):
        """Allow loops to be composed in pipelines."""
        return Pipeline(nodes=[self, other])

    def to_dict(self) -> dict:
        """Export loop structure."""
        return {
            "type": "Loop",
            "name": self.name,
            "max_iterations": self.max_iterations,
            "condition": self.condition.__name__ if hasattr(self.condition, '__name__') else "lambda",
            "body": (
                self.body.to_dict() if hasattr(self.body, 'to_dict')
                else {"type": "Unknown", "name": str(self.body)}
            )
        }

    def __repr__(self) -> str:
        return f"Loop(name='{self.name}', max_iterations={self.max_iterations})"

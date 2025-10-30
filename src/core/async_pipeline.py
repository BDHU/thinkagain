"""
Async versions of Pipeline components for async/await workflows.

Provides AsyncWorker, AsyncPipeline, AsyncSwitch, and AsyncLoop that work
with async/await syntax while inheriting behavior from sync counterparts.
"""

import asyncio
from typing import List
from .pipeline import Pipeline, Switch, Loop
from .worker import Worker
from .context import Context


class AsyncWorker(Worker):
    """
    Base class for async workers - handles async operations.

    Async workers use async/await for non-blocking operations like:
    - Async HTTP requests
    - Async database queries
    - Async LLM API calls
    - Async file I/O

    Example:
        class AsyncVectorSearch(AsyncWorker):
            async def __call__(self, ctx: Context) -> Context:
                ctx.documents = await self.db.search_async(ctx.query)
                ctx.log(f"[{self.name}] Retrieved {len(ctx.documents)} docs")
                return ctx
    """

    async def __call__(self, ctx: Context) -> Context:
        """
        Process context asynchronously.

        Args:
            ctx: Input context

        Returns:
            Modified context

        Raises:
            NotImplementedError: Subclasses must implement this
        """
        raise NotImplementedError(
            f"AsyncWorker '{self.name}' must implement async __call__"
        )

    def __rshift__(self, other):
        """
        Compose async workers into AsyncPipeline using >> operator.

        Example:
            pipeline = async_worker1 >> async_worker2 >> async_worker3
        """
        if isinstance(other, AsyncPipeline):
            return AsyncPipeline(nodes=[self] + other.nodes)
        # Always return AsyncPipeline, not Pipeline
        return AsyncPipeline(nodes=[self, other])

    # Inherits: __init__, _generate_name, to_dict, __repr__


class AsyncPipeline(Pipeline):
    """
    Async pipeline - executes async workers sequentially.

    Each worker receives the context from the previous worker.
    All execution is non-blocking using async/await.

    Example:
        pipeline = AsyncPipeline([worker1, worker2, worker3])
        result = await pipeline(ctx)

        # Or run synchronously:
        result = pipeline.run(ctx)
    """

    async def __call__(self, ctx: Context) -> Context:
        """Execute async nodes in sequence."""
        ctx.log(f"[AsyncPipeline] Starting: {self.name}")

        for node in self.nodes:
            ctx = await node(ctx)

        ctx.log(f"[AsyncPipeline] Completed: {self.name}")
        return ctx

    def run(self, ctx: Context) -> Context:
        """
        Convenience method to run async pipeline synchronously.

        This wraps asyncio.run() so you can call pipeline.run(ctx)
        instead of asyncio.run(pipeline(ctx)).
        """
        return asyncio.run(self(ctx))

    def __rshift__(self, other):
        """Extend async pipeline with another async worker/pipeline."""
        if isinstance(other, AsyncPipeline):
            return AsyncPipeline(
                nodes=self.nodes + other.nodes,
                name=f"{self.name}_extended"
            )
        return AsyncPipeline(
            nodes=self.nodes + [other],
            name=f"{self.name}_extended"
        )

    # Inherits: __init__, to_dict, visualize, __repr__


class AsyncSwitch(Switch):
    """
    Async multi-way conditional branching.

    Evaluates conditions in order and executes the first matching branch.
    All branches are async workers.

    Example:
        switch = AsyncSwitch(name="quality_check") \
            .case(lambda ctx: len(ctx.documents) >= 3, high_quality_branch) \
            .case(lambda ctx: len(ctx.documents) >= 1, medium_branch) \
            .set_default(fallback_branch)

        pipeline = retriever >> switch >> generator
        result = await pipeline(ctx)
    """

    async def __call__(self, ctx: Context) -> Context:
        """Evaluate conditions and execute first matching async branch."""
        ctx.log(f"[AsyncSwitch] Evaluating: {self.name}")

        for i, (condition, branch) in enumerate(self.cases):
            # Allow async conditions
            if asyncio.iscoroutinefunction(condition):
                result = await condition(ctx)
            else:
                result = condition(ctx)

            ctx.log(f"[AsyncSwitch] Case {i+1} evaluated to: {result}")

            if result:
                ctx.log(f"[AsyncSwitch] Executing case {i+1}")
                return await branch(ctx)

        # Default branch
        if self.default:
            ctx.log(f"[AsyncSwitch] Executing default")
            return await self.default(ctx)

        ctx.log(f"[AsyncSwitch] No match, passing through")
        return ctx

    def __rshift__(self, other):
        """Allow async switches to be composed in async pipelines."""
        return AsyncPipeline(nodes=[self, other])

    # Inherits: __init__, case(), set_default(), to_dict(), __repr__


class AsyncLoop(Loop):
    """
    Async loop - repeatedly executes async body while condition is true.

    Example:
        loop = AsyncLoop(
            condition=lambda ctx: len(ctx.documents) < 2,
            body=query_refiner >> retriever,
            max_iterations=3
        )

        pipeline = retriever >> loop >> generator
        result = await pipeline(ctx)
    """

    async def __call__(self, ctx: Context) -> Context:
        """Execute async body repeatedly while condition is true."""
        iteration = 0
        ctx.log(f"[AsyncLoop] Starting: {self.name}")

        while iteration < self.max_iterations:
            # Allow async conditions
            if asyncio.iscoroutinefunction(self.condition):
                should_continue = await self.condition(ctx)
            else:
                should_continue = self.condition(ctx)

            if not should_continue:
                break

            ctx.log(f"[AsyncLoop] Iteration {iteration + 1}/{self.max_iterations}")
            ctx = await self.body(ctx)
            iteration += 1

        if iteration >= self.max_iterations:
            ctx.log(f"[AsyncLoop] Terminated: max iterations reached")
        else:
            ctx.log(f"[AsyncLoop] Completed after {iteration} iterations")

        return ctx

    def __rshift__(self, other):
        """Allow async loops to be composed in async pipelines."""
        return AsyncPipeline(nodes=[self, other])

    # Inherits: __init__, to_dict(), __repr__

"""
Parallel execution patterns for async pipelines.

Provides Parallel for running multiple workers concurrently,
useful when multiple nodes depend on the same parent node.
"""

import asyncio
from typing import List, Callable, Optional
from .context import Context
from .worker import Worker
from .runtime import _invoke


class Parallel(Worker):
    """
    Execute multiple async workers in parallel with the same input.

    This is useful when you have multiple nodes that depend on the same
    parent node and can execute concurrently.

    Example:
        # Both vector search and web search depend on query refiner
        #
        #       query_refiner
        #          /      \\
        #   vector_db    web_search   (run in parallel)
        #          \\      /
        #          reranker
        #
        pipeline = (
            query_refiner
            >> Parallel([vector_db, web_search])
            >> reranker
            >> generator
        )

    The default behavior is to merge all documents from parallel workers.
    You can customize this by providing a merge_strategy function.

    Custom merge example:
        def custom_merge(original_ctx, results):
            merged = original_ctx.copy()
            # Keep only results from first worker
            merged.documents = results[0].documents if results else []
            return merged

        parallel = Parallel(
            workers=[worker1, worker2],
            merge_strategy=custom_merge
        )
    """

    def __init__(
        self,
        workers: List[Worker],
        merge_strategy: Optional[Callable[[Context, List[Context]], Context]] = None,
        name: str = "parallel",
    ):
        """
        Initialize parallel execution.

        Args:
            workers: List of workers to run in parallel
            merge_strategy: Optional function to merge results.
                           Signature: (original_ctx, results) -> merged_ctx
                           Default: combines all documents from workers
            name: Name for this component (for logging)
        """
        super().__init__(name=name)
        self.workers = workers
        self.merge_strategy = merge_strategy or self._default_merge

    async def arun(self, ctx: Context) -> Context:
        """Execute all workers concurrently and merge their results."""
        self._log(ctx, f"Starting {len(self.workers)} workers in parallel")

        tasks = [self._execute_worker(worker, ctx.copy()) for worker in self.workers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                worker_name = getattr(self.workers[idx], "name", f"worker_{idx}")
                self._log(ctx, f"{worker_name} failed: {result}")
            else:
                successful_results.append(result)

        if not successful_results:
            self._log(ctx, "All workers failed, returning original context")
            return ctx

        merged_ctx = self.merge_strategy(ctx, successful_results)
        self._log(ctx, "Completed parallel execution")
        return merged_ctx

    def _default_merge(self, original_ctx: Context, results: List[Context]) -> Context:
        """
        Default merge strategy: combine all documents from parallel workers.

        Takes documents from all successful workers and concatenates them
        into a single list. Other context fields from the original context
        are preserved.

        Args:
            original_ctx: Original input context
            results: List of result contexts from successful workers

        Returns:
            Merged context with combined documents
        """
        merged = original_ctx.copy()
        all_documents = []

        for result in results:
            if hasattr(result, "documents") and result.documents:
                all_documents.extend(result.documents)

        merged.documents = all_documents
        merged.log(
            f"[{self.name}] Merged {len(all_documents)} documents from {len(results)} workers"
        )
        return merged

    async def _execute_worker(self, worker: Worker, ctx: Context) -> Context:
        """Run a worker that may only provide sync or async APIs."""
        return await _invoke(worker, ctx)

    def _log(self, ctx: Context, message: str) -> None:
        ctx.log(f"[{self.name}] {message}")

    def __repr__(self) -> str:
        return f"Parallel(name='{self.name}', workers={len(self.workers)})"

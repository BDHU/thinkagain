"""
Parallel execution patterns for async pipelines.

Provides Parallel for running multiple workers concurrently,
useful when multiple nodes depend on the same parent node.
"""

import asyncio
from typing import List, Callable, Optional
from .context import Context
from .worker import Worker


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
        name: str = "parallel"
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

    async def acall(self, ctx: Context) -> Context:
        """Execute all workers in parallel with same input context (async only)."""
        ctx.log(f"[{self.name}] Starting {len(self.workers)} workers in parallel")

        # Each worker gets a copy of the input context
        # Call acall() if available, otherwise call __call__
        tasks = []
        for worker in self.workers:
            if hasattr(worker, 'acall'):
                tasks.append(worker.acall(ctx.copy()))
            else:
                tasks.append(worker(ctx.copy()))

        # Execute all workers concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                worker_name = self.workers[i].name if hasattr(self.workers[i], 'name') else f"worker_{i}"
                ctx.log(f"[{self.name}] {worker_name} failed: {result}")

        # Filter out exceptions, keep only successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]

        if not successful_results:
            ctx.log(f"[{self.name}] All workers failed, returning original context")
            return ctx

        # Merge results using strategy
        merged_ctx = self.merge_strategy(ctx, successful_results)

        ctx.log(f"[{self.name}] Completed parallel execution")
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
            if hasattr(result, 'documents') and result.documents:
                all_documents.extend(result.documents)

        merged.documents = all_documents
        merged.log(f"[{self.name}] Merged {len(all_documents)} documents from {len(results)} workers")
        return merged

    def to_dict(self) -> dict:
        """Export parallel structure as dictionary."""
        return {
            "type": "Parallel",
            "name": self.name,
            "workers": [
                worker.to_dict() if hasattr(worker, 'to_dict')
                else {"type": "Unknown", "name": str(worker)}
                for worker in self.workers
            ]
        }

    def __repr__(self) -> str:
        return f"Parallel(name='{self.name}', workers={len(self.workers)})"

import re
from typing import Optional, Callable, List, Union, TYPE_CHECKING
from .context import Context

if TYPE_CHECKING:
    from .pipeline import Pipeline


class Worker:
    """
    Base class for workers in the agent framework.

    Workers are units of computation that:
    - Take a Context as input
    - Perform some operation
    - Return the modified Context
    - Log their actions for debugging
    - Can be composed using >> operator

    Workers can be simple functions, complex stateful services,
    or anything in between (e.g., vector DB, LLM, reranker).

    Example:
        class VectorDBWorker(Worker):
            def __call__(self, ctx: Context) -> Context:
                ctx.documents = self.search(ctx.query)
                ctx.log(f"[{self.name}] Retrieved {len(ctx.documents)} docs")
                return ctx

        # Compose workers into pipelines
        pipeline = vector_db >> reranker >> generator
        ctx = pipeline(Context(query="What is ML?"))
    """

    def __init__(self, name: str = None):
        """
        Initialize worker with a name.

        Args:
            name: Identifier for this worker (used in logging).
                  If not provided, auto-generates from class name.
        """
        self.name = name or self._generate_name()

    def _generate_name(self) -> str:
        """
        Generate a name from the class name.

        Examples:
            VectorDBWorker -> vector_db_worker
            RerankerWorker -> reranker_worker
        """
        class_name = self.__class__.__name__
        # Convert CamelCase to snake_case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return name

    def __call__(self, ctx: Context) -> Context:
        """
        Process the context and return modified context.

        Args:
            ctx: Input context

        Returns:
            Modified context

        Raises:
            NotImplementedError: Subclasses must implement this
        """
        raise NotImplementedError(f"Worker '{self.name}' must implement __call__")

    def __rshift__(self, other: 'Worker') -> 'Pipeline':
        """
        Compose workers using >> operator.

        Args:
            other: Next worker/pipeline in the chain

        Returns:
            Pipeline containing both workers

        Example:
            pipeline = worker1 >> worker2 >> worker3
        """
        # Import at runtime to avoid circular dependency (Pipeline imports Worker)
        from .pipeline import Pipeline

        if isinstance(other, Pipeline):
            return Pipeline(nodes=[self] + other.nodes)
        return Pipeline(nodes=[self, other])

    def to_dict(self) -> dict:
        """Export worker structure as dictionary."""
        return {
            "type": "Worker",
            "name": self.name,
            "class": self.__class__.__name__
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
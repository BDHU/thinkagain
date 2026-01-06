"""Sequential container for composing node pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    from .context import Context


def _step_name(step: Callable) -> str:
    if hasattr(step, "name"):
        return step.name
    if hasattr(step, "__name__"):
        return step.__name__
    return str(step)


class Sequential:
    """Sequential container for composing nodes into a pipeline.

    Similar to PyTorch's nn.Sequential, this allows declarative pipeline
    definition that can be reused, composed, and inspected.

    Example:
        @node
        async def retrieve(s: State) -> State:
            return State(docs=["doc1", "doc2"])

        @node
        async def generate(s: State) -> State:
            return State(answer="Answer")

        # Define reusable pipeline
        pipeline = Sequential(
            retrieve,
            generate,
        )

        # Use anywhere
        result = run(pipeline, State())

        # Compose pipelines
        full_pipeline = Sequential(
            preprocessing_pipeline,
            main_pipeline,
            postprocessing_pipeline,
        )
    """

    __slots__ = ("steps",)

    def __init__(self, *steps: Callable[["Context"], "Context"] | Sequential):
        """Initialize with a sequence of nodes or nested Sequential containers.

        Args:
            *steps: Node functions or Sequential containers to chain together.
        """
        # Flatten nested Sequential containers
        flattened = []
        for step in steps:
            if isinstance(step, Sequential):
                flattened.extend(step.steps)
            else:
                flattened.append(step)
        self.steps = tuple(flattened)

    def __call__(self, ctx: "Context") -> "Context":
        """Execute the pipeline by chaining all steps.

        Args:
            ctx: Input context to pass through the pipeline.

        Returns:
            Output context after all steps have been applied.
        """
        for step in self.steps:
            ctx = step(ctx)
        return ctx

    def __repr__(self) -> str:
        """Return string representation showing pipeline structure."""
        if not self.steps:
            return "Sequential()"

        step_names = self.node_names
        if len(step_names) <= 3:
            return f"Sequential({', '.join(step_names)})"
        return f"Sequential({step_names[0]}, ..., {step_names[-1]}) [{len(step_names)} steps]"

    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, idx: int) -> Callable[["Context"], "Context"]:
        """Get a specific step by index."""
        return self.steps[idx]

    def __iter__(self) -> Iterator[Callable[["Context"], "Context"]]:
        """Iterate over pipeline steps."""
        return iter(self.steps)

    @property
    def node_names(self) -> list[str]:
        """Get names of all nodes in the pipeline."""
        return [_step_name(step) for step in self.steps]

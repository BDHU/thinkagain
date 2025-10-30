"""
Minimal Agent Framework

A simple, debuggable framework for building agent pipelines with explicit control.

Key principles:
- Explicit over implicit
- Full transparency and inspectability
- Easy to debug at every step
- Minimal abstractions

Core Components:
- Context: Holds state and history as it passes through workers
- Worker: Base class for any processing unit (vector DB, LLM, etc.)
- Pipeline: Compose workers with >> operator and control flow

Example RAG Pipeline:
    from src.core import Context, Worker, Pipeline, Switch

    # Option 1: Functional composition
    pipeline = vector_db >> reranker >> generator
    ctx = pipeline.run(Context(query="What is ML?"))

    # Option 2: With control flow
    pipeline = (
        vector_db
        >> Switch(name="quality_check")
            .case(lambda ctx: len(ctx.documents) >= 2, reranker)
            .set_default(fallback >> reranker)
        >> generator
    )

    # Graph structure is captured before execution
    print(pipeline.visualize())

    # Execute and inspect
    ctx = pipeline.run(Context(query="What is ML?"))
    print(ctx.answer)
    print(ctx.history)
"""

from .core import Context, Worker, Pipeline, Conditional, Switch, Loop

__version__ = "0.2.0"

__all__ = [
    "Context",
    "Worker",
    "Pipeline",
    "Conditional",
    "Switch",
    "Loop",
]

"""
RAG Pipeline with Computation Graph Example

Demonstrates how to build a RAG pipeline using the functional DSL with:
- Operator composition (>>)
- Conditional branching (if/elif/else)
- Loops
- Graph visualization and export
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Context, Conditional, Switch, Loop
from shared_workers import (
    VectorDBWorker,
    WebSearchWorker,
    RerankerWorker,
    QueryRefinerWorker,
    GeneratorWorker
)


# ============================================================================
# Example 1: Simple Linear Pipeline
# ============================================================================

def example_simple_pipeline():
    print("=" * 80)
    print("EXAMPLE 1: Simple Linear Pipeline")
    print("=" * 80)
    print()

    # Create workers
    vector_db = VectorDBWorker()
    reranker = RerankerWorker()
    generator = GeneratorWorker()

    # Build pipeline using >> operator
    pipeline = vector_db >> reranker >> generator

    # Inspect graph BEFORE execution
    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    print("Pipeline as Dict:")
    print(json.dumps(pipeline.to_dict(), indent=2))
    print()

    # Execute
    ctx = Context(query="What is machine learning?", top_k=3, rerank_top_n=2)
    result = pipeline.run(ctx)

    print("Result:")
    print(f"  Answer: {result.answer}")
    print()

    print("Execution History:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Example 2: Pipeline with Binary Conditional
# ============================================================================

def example_conditional_pipeline():
    print("=" * 80)
    print("EXAMPLE 2: Pipeline with Conditional")
    print("=" * 80)
    print()

    # Create workers
    vector_db = VectorDBWorker()
    web_search = WebSearchWorker()
    reranker = RerankerWorker()
    generator = GeneratorWorker()

    # Build pipeline with conditional
    pipeline = (
        vector_db
        >> Conditional(
            condition=lambda ctx: len(ctx.documents) >= 2,
            true_branch=reranker,
            false_branch=web_search >> reranker,
            name="check_document_count"
        )
        >> generator
    )

    # Visualize graph
    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Execute with query that returns few docs
    ctx = Context(query="quantum computing", top_k=3)
    result = pipeline.run(ctx)

    print("Execution History:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Example 3: Multi-way Conditional (Switch)
# ============================================================================

def example_switch_pipeline():
    print("=" * 80)
    print("EXAMPLE 3: Multi-way Conditional (Switch)")
    print("=" * 80)
    print()

    # Create workers
    vector_db = VectorDBWorker()
    web_search = WebSearchWorker()
    reranker = RerankerWorker()
    generator = GeneratorWorker()

    # Build pipeline with switch (if/elif/else)
    pipeline = (
        vector_db
        >> Switch(name="quality_routing")
            .case(lambda ctx: len(ctx.documents) >= 5, reranker)  # High quality
            .case(lambda ctx: len(ctx.documents) >= 2, reranker)  # Medium quality
            .set_default(web_search >> reranker)  # Low quality - add web search
        >> generator
    )

    # Alternative: using Switch constructor directly
    # pipeline = (
    #     vector_db
    #     >> Switch(
    #         cases=[
    #             (lambda ctx: len(ctx.documents) >= 5, high_quality_path),
    #             (lambda ctx: len(ctx.documents) >= 2, medium_quality_path),
    #         ],
    #         default=low_quality_path,
    #         name="quality_routing"
    #     )
    #     >> generator
    # )

    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    print("Pipeline as Dict:")
    print(json.dumps(pipeline.to_dict(), indent=2))
    print()

    # Execute
    ctx = Context(query="What is AI?", top_k=3)
    result = pipeline.run(ctx)

    print("Execution History:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Example 4: Pipeline with Loop
# ============================================================================

def example_loop_pipeline():
    print("=" * 80)
    print("EXAMPLE 4: Pipeline with Loop")
    print("=" * 80)
    print()

    # Create workers
    vector_db = VectorDBWorker()
    query_refiner = QueryRefinerWorker()
    generator = GeneratorWorker()

    # Build pipeline with retry loop
    pipeline = (
        vector_db
        >> Loop(
            condition=lambda ctx: len(ctx.documents) < 2 and ctx.get("retry_count", 0) < 2,
            body=query_refiner >> vector_db,
            max_iterations=3,
            name="retry_retrieval"
        )
        >> generator
    )

    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Execute with query that returns few docs initially
    ctx = Context(query="xyz", top_k=3, retry_count=0)
    result = pipeline.run(ctx)

    print("Execution History:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Example 5: Complex Nested Pipeline
# ============================================================================

def example_complex_pipeline():
    print("=" * 80)
    print("EXAMPLE 5: Complex Nested Pipeline")
    print("=" * 80)
    print()

    # Create workers
    vector_db = VectorDBWorker()
    web_search = WebSearchWorker()
    reranker = RerankerWorker()
    query_refiner = QueryRefinerWorker()
    generator = GeneratorWorker()

    # Build complex pipeline with nested control flow
    retrieval_with_retry = (
        vector_db
        >> Loop(
            condition=lambda ctx: len(ctx.documents) < 2,
            body=query_refiner >> vector_db,
            max_iterations=2,
            name="retry_loop"
        )
    )

    quality_check = Switch(name="quality_check") \
        .case(lambda ctx: len(ctx.documents) >= 3, reranker) \
        .set_default(web_search >> reranker)

    pipeline = retrieval_with_retry >> quality_check >> generator

    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    print("Pipeline as Dict:")
    print(json.dumps(pipeline.to_dict(), indent=2))
    print()

    # Execute
    ctx = Context(query="What is deep learning?", top_k=3)
    result = pipeline.run(ctx)

    print("Final Answer:")
    print(f"  {result.answer}")
    print()

    print("Full Execution Trace:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    example_simple_pipeline()
    print("\n")

    example_conditional_pipeline()
    print("\n")

    example_switch_pipeline()
    print("\n")

    example_loop_pipeline()
    print("\n")

    example_complex_pipeline()

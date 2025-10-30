"""
Simple RAG (Retrieval-Augmented Generation) Pipeline Example

This demonstrates the minimal agent framework with explicit control
and full debuggability. Each step is transparent and inspectable.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.context import Context
from shared_workers import VectorDBWorker, RerankerWorker, GeneratorWorker


# ============================================================================
# RAG Pipeline Execution
# ============================================================================

def run_rag_pipeline(query: str, verbose: bool = True):
    """
    Run a complete RAG pipeline with full visibility and control.

    The pipeline:
    1. Retrieve relevant documents from vector DB
    2. Rerank documents for better relevance
    3. Generate answer using retrieved context

    Args:
        query: User's question
        verbose: Whether to print intermediate steps

    Returns:
        Final context with answer
    """
    # Initialize context with query and parameters
    ctx = Context(
        query=query,
        top_k=5,          # Retrieve top 5 documents
        rerank_top_n=2    # Keep top 2 after reranking
    )

    if verbose:
        print(f"Starting RAG pipeline for query: '{query}'\n")

    # Initialize workers
    vector_db = VectorDBWorker(collection="ml_knowledge")
    reranker = RerankerWorker()
    generator = GeneratorWorker(model="gpt-4")

    # Step 1: Retrieval
    ctx = vector_db(ctx)
    if verbose:
        print(f"Step 1 - Retrieval:")
        print(f"  Retrieved {len(ctx.documents)} documents")
        for i, doc in enumerate(ctx.documents, 1):
            print(f"    {i}. {doc[:80]}...")
        print()

    # Step 2: Reranking
    ctx = reranker(ctx)
    if verbose:
        print(f"Step 2 - Reranking:")
        print(f"  Reranked to {len(ctx.documents)} documents")
        for i, doc in enumerate(ctx.documents, 1):
            print(f"    {i}. {doc[:80]}...")
        print()

    # Step 3: Generation
    ctx = generator(ctx)
    if verbose:
        print(f"Step 3 - Generation:")
        print(f"  Prompt length: {len(ctx.prompt)} chars")
        print(f"  Answer: {ctx.answer}")
        print()

    # Show full execution history
    if verbose:
        print("Execution History:")
        for log in ctx.history:
            print(f"  {log}")
        print()

    return ctx


# ============================================================================
# Alternative: Step-by-step Execution (Maximum Control)
# ============================================================================

def run_rag_step_by_step(query: str):
    """
    Demonstrates running the pipeline step-by-step with inspection
    between each step. This gives maximum control and debuggability.
    """
    print(f"Query: {query}\n")

    # Initialize
    ctx = Context(query=query, top_k=3, rerank_top_n=2)
    vector_db = VectorDBWorker()
    reranker = RerankerWorker()
    generator = GeneratorWorker()

    # Step 1: Retrieve
    print(">>> Running retrieval...")
    ctx = vector_db(ctx)
    print(f"Retrieved documents: {ctx.documents}")
    print(f"Should we continue? (In real app, you could inspect and decide)\n")

    # Step 2: Rerank
    print(">>> Running reranking...")
    ctx = reranker(ctx)
    print(f"Reranked documents: {ctx.documents}")
    print(f"Looks good? (You have full control here)\n")

    # Step 3: Generate
    print(">>> Running generation...")
    ctx = generator(ctx)
    print(f"Prompt used:\n{ctx.prompt}\n")
    print(f"Answer: {ctx.answer}\n")

    return ctx


# ============================================================================
# Alternative: Functional Composition
# ============================================================================

def run_rag_functional(query: str):
    """
    Demonstrates functional composition style.
    Clean and concise while maintaining full transparency.
    """
    # Create workers
    vector_db = VectorDBWorker()
    reranker = RerankerWorker()
    generator = GeneratorWorker()

    # Chain workers - each step is explicit
    ctx = Context(query=query, top_k=3, rerank_top_n=2)
    ctx = generator(reranker(vector_db(ctx)))

    # Still fully inspectable!
    print(f"Answer: {ctx.answer}")
    print(f"History: {ctx.history}")

    return ctx


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Simple RAG Pipeline Example")
    print("=" * 80)
    print()

    # Example 1: Full pipeline with verbose output
    print("EXAMPLE 1: Full Pipeline\n")
    result = run_rag_pipeline("What is machine learning?", verbose=True)

    print("\n" + "=" * 80 + "\n")

    # Example 2: Step-by-step execution
    print("EXAMPLE 2: Step-by-Step Execution\n")
    result = run_rag_step_by_step("Explain transformers")

    print("\n" + "=" * 80 + "\n")

    # Example 3: Functional composition
    print("EXAMPLE 3: Functional Composition\n")
    result = run_rag_functional("What are embeddings?")

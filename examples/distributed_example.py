"""Comprehensive distributed execution example with thinkagain.

This example demonstrates the complete distributed API:
1. @replica decorator for distributed classes
2. @node decorator for pure functions
3. @jit decorator for compiling pipelines
4. Mesh for defining execution resources
5. Local and remote execution patterns

Usage:
    # LOCAL EXECUTION (no servers needed):
    python examples/distributed_example.py

    # REMOTE EXECUTION (with gRPC servers):
    # Terminal 1 - Start LLM server:
    python -m thinkagain.serve examples.distributed_example:LLM --port 8000

    # Terminal 2 - Start Retriever server:
    python -m thinkagain.serve examples.distributed_example:Retriever --port 8001

    # Terminal 3 - Run client with remote mesh:
    python examples/distributed_example.py --remote

Key Concepts:
- @replica: Marks stateful classes for distributed execution
- @node: Marks pure functions for graph compilation
- @jit: Compiles pipelines for optimization and distribution
- Mesh: Defines available computational resources
- MeshNode: Represents a single compute node (local or remote)
"""

import argparse
import asyncio

import thinkagain as ta


# ============================================================================
# Mock LLM Engine (for demonstration)
# ============================================================================


class MockEngine:
    """Mock LLM engine simulating a real inference engine."""

    def __init__(self, model_name: str):
        print(f"  [INIT] Loading model: {model_name}")
        self.model_name = model_name

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        await asyncio.sleep(0.1)  # Simulate inference
        return f"[{self.model_name}@temp={temperature}]: Response to '{prompt[:50]}...'"


# ============================================================================
# Define Replicas
# ============================================================================


@ta.replica()  # CPU-only replica
class Retriever:
    """Document retriever - can scale freely on CPU.

    This replica demonstrates:
    - CPU-only workloads (no GPU requirement)
    - Lightweight initialization
    - Stateless operations (though stateful is supported)
    """

    def __init__(self):
        print("  [INIT] Retriever initialized")

    async def __call__(self, query: str) -> list[str]:
        """Retrieve documents for a query."""
        await asyncio.sleep(0.02)  # Simulate database query
        return [f"Document {i}: Information about '{query}'" for i in range(1, 4)]


@ta.replica(gpus=1, backend="grpc")  # GPU replica with gRPC backend
class LLM:
    """Stateful LLM replica - heavy initialization, GPU-bound.

    This replica demonstrates:
    - GPU resource requirements (gpus=1)
    - Heavy initialization (model loading)
    - Stateful operations
    - Remote serving with gRPC backend
    """

    def __init__(self, model: str = "llama-70b"):
        # Heavy initialization (loads model into GPU memory)
        self.engine = MockEngine(model)

    async def __call__(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        return await self.engine.generate(prompt, temperature)


@ta.replica(backend="grpc")  # Simple text processing replica
class TextProcessor:
    """Stateful text processor for various transformations.

    This replica demonstrates:
    - Maintaining state across calls (request counter)
    - Multiple operation types
    - Simple CPU-bound processing
    """

    def __init__(self):
        self.request_count = 0
        print("  [INIT] TextProcessor initialized")

    async def __call__(self, text: str, operation: str = "upper") -> str:
        """Process text with various operations.

        Args:
            text: Input string
            operation: One of "upper", "lower", "title", "reverse", "swap"

        Returns:
            Processed string
        """
        self.request_count += 1

        operations = {
            "upper": text.upper,
            "lower": text.lower,
            "title": text.title,
            "reverse": lambda: text[::-1],
            "swap": text.swapcase,
        }

        result = operations.get(operation, lambda: text)()
        print(
            f"  [Server] Request #{self.request_count}: {operation}('{text}') -> '{result}'"
        )
        return result


# ============================================================================
# Define Nodes (Pure Functions)
# ============================================================================


@ta.node
async def format_rag_prompt(query: str, docs: list[str]) -> str:
    """Format a RAG prompt from query and documents.

    This node demonstrates:
    - Pure function decoration with @node
    - Simple data transformation
    - Used within @jit pipelines
    """
    context = "\n".join(f"- {doc}" for doc in docs[:3])
    return f"""Context:
{context}

Query: {query}

Answer:"""


@ta.node
async def combine_results(original: str, *processed: str) -> dict:
    """Combine processing results into a dictionary.

    Args:
        original: Original input text
        *processed: Processed results

    Returns:
        Dictionary with all results
    """
    return {"original": original, "results": list(processed)}


# ============================================================================
# Create Replica Handles (Outside @jit)
# ============================================================================

# These handles are used within @jit pipelines
retriever = Retriever.init()  # type: ignore[attr-defined]
llm = LLM.init("llama-70b")  # type: ignore[attr-defined]
text_processor = TextProcessor.init()  # type: ignore[attr-defined]


# ============================================================================
# Define Pipelines with @jit
# ============================================================================


@ta.jit
async def rag_pipeline(query: str) -> str:
    """RAG (Retrieval-Augmented Generation) pipeline.

    This pipeline demonstrates:
    - Sequential operations (retrieve -> format -> generate)
    - Mixing replicas (@replica) and nodes (@node)
    - Automatic optimization within @jit

    Flow:
    1. Retrieve relevant documents
    2. Format prompt with documents
    3. Generate response with LLM
    """
    # Step 1: Retrieve documents
    docs = await retriever(query)

    # Step 2: Format prompt
    prompt = await format_rag_prompt(query, docs)

    # Step 3: Generate response
    response = await llm(prompt, temperature=0.7)

    return response


@ta.jit
async def text_processing_pipeline(text: str, operations: list[str]) -> dict:
    """Multi-operation text processing pipeline.

    This pipeline demonstrates:
    - Parallel operations (all transformations can run concurrently)
    - Using __call__ method on replicas
    - Explicit operation parameters (clean API!)

    Args:
        text: Input text to process
        operations: List of operations to apply (e.g., ["upper", "lower", "title"])

    Flow:
    1. Apply multiple text transformations in parallel
    2. Collect all results
    """
    # Create result dict starting with original
    result = {"original": text}

    # Apply each operation (can potentially run in parallel)
    for operation in operations:
        result[operation] = await text_processor(text, operation=operation)

    return result


@ta.jit
async def complex_rag_pipeline(query: str) -> dict:
    """Complex RAG pipeline with post-processing.

    This pipeline demonstrates:
    - Mixing both RAG and text processing
    - More complex data flow
    - Reusing other pipelines (composability)
    """
    # Get RAG response
    response = await rag_pipeline(query)

    # Post-process the response
    processed = await text_processor(response, operation="title")

    return {
        "query": query,
        "response": response,
        "processed_response": processed,
    }


# ============================================================================
# Example Execution Functions
# ============================================================================


async def run_local_example():
    """Run examples with local mesh (no remote servers needed)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: LOCAL EXECUTION")
    print("=" * 70)
    print("\nRunning pipelines with local mesh (single GPU node)...")

    # Create local mesh with 1 GPU
    mesh = ta.Mesh([ta.MeshNode("local", gpus=1)])

    with mesh:
        # Example 1a: Single RAG query
        print("\n[1a] Single RAG Query:")
        result = await rag_pipeline("What is machine learning?")
        print("  Query: What is machine learning?")
        print(f"  Response: {result}")

        # Example 1b: Multiple RAG queries in parallel
        print("\n[1b] Multiple RAG Queries (parallel):")
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "What are transformers?",
        ]
        results = await asyncio.gather(*[rag_pipeline(q) for q in queries])
        for q, r in zip(queries, results):
            print(f"  Q: {q}")
            print(f"  A: {r}")

        # Example 1c: Text processing pipeline
        print("\n[1c] Text Processing Pipeline:")
        text = "Hello Distributed World"
        operations = ["upper", "lower", "title", "reverse"]
        result = await text_processing_pipeline(text, operations)
        print(f"  Original: {result['original']}")
        print(f"  Upper: {result['upper']}")
        print(f"  Lower: {result['lower']}")
        print(f"  Title: {result['title']}")
        print(f"  Reverse: {result['reverse']}")

        # Example 1d: Complex pipeline
        print("\n[1d] Complex RAG Pipeline:")
        result = await complex_rag_pipeline("How do neural networks learn?")
        print(f"  Query: {result['query']}")
        print(f"  Response: {result['response']}")
        print(f"  Processed: {result['processed_response']}")


async def run_remote_example():
    """Run examples with remote mesh (requires gRPC servers)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: REMOTE EXECUTION")
    print("=" * 70)
    print("\nRunning pipelines with remote mesh (gRPC servers)...")
    print("\nMake sure you've started the servers:")
    print(
        "  Terminal 1: python -m thinkagain.serve examples.distributed_example:LLM --port 8000"
    )
    print(
        "  Terminal 2: python -m thinkagain.serve examples.distributed_example:Retriever --port 8001"
    )
    print(
        "  Terminal 3: python -m thinkagain.serve examples.distributed_example:TextProcessor --port 8002"
    )

    # Create remote mesh with gRPC endpoints
    mesh = ta.Mesh(
        [
            ta.MeshNode("llm_server", endpoint="localhost:8000", gpus=1),
            ta.MeshNode("retriever_server", endpoint="localhost:8001"),
            ta.MeshNode("processor_server", endpoint="localhost:8002"),
        ]
    )

    try:
        with mesh:
            # Example 2a: RAG with remote services
            print("\n[2a] RAG Pipeline (remote execution):")
            result = await rag_pipeline("What is distributed computing?")
            print("  Query: What is distributed computing?")
            print(f"  Response: {result}")

            # Example 2b: Text processing with remote service
            print("\n[2b] Text Processing (remote execution):")
            text = "Remote Processing Test"
            operations = ["upper", "lower", "reverse"]
            result = await text_processing_pipeline(text, operations)
            print(f"  Original: {result['original']}")
            print(f"  Upper: {result['upper']}")
            print(f"  Lower: {result['lower']}")
            print(f"  Reverse: {result['reverse']}")

            # Example 2c: Multiple queries demonstrating load balancing
            print("\n[2c] Multiple Queries (demonstrates remote execution):")
            queries = ["Query 1", "Query 2", "Query 3"]
            results = await asyncio.gather(*[rag_pipeline(q) for q in queries])
            for i, (q, r) in enumerate(zip(queries, results), 1):
                print(f"  {i}. {q}: {r}")

    except Exception as e:
        print(f"\n❌ Error connecting to remote servers: {e}")
        print("\nMake sure servers are running:")
        print(
            "  python -m thinkagain.serve examples.distributed_example:LLM --port 8000"
        )
        print(
            "  python -m thinkagain.serve examples.distributed_example:Retriever --port 8001"
        )
        print(
            "  python -m thinkagain.serve examples.distributed_example:TextProcessor --port 8002"
        )


async def run_hybrid_example():
    """Run examples with hybrid mesh (mix of local and remote)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: HYBRID EXECUTION")
    print("=" * 70)
    print("\nRunning pipelines with hybrid mesh (local + remote)...")

    # Create hybrid mesh: local retriever, remote LLM
    mesh = ta.Mesh(
        [
            ta.MeshNode("local", gpus=0),  # Local CPU node
            ta.MeshNode("remote_llm", endpoint="localhost:8000", gpus=1),  # Remote GPU
        ]
    )

    try:
        with mesh:
            print("\n[3a] Hybrid RAG (local retriever, remote LLM):")
            result = await rag_pipeline("Explain hybrid execution")
            print(f"  Response: {result}")

    except Exception as e:
        print(f"\n❌ Error in hybrid execution: {e}")
        print("\nNote: Start LLM server for this example:")
        print(
            "  python -m thinkagain.serve examples.distributed_example:LLM --port 8000"
        )


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Thinkagain Distributed Execution Examples"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run remote execution examples (requires gRPC servers)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Run hybrid execution examples (mix of local and remote)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("THINKAGAIN DISTRIBUTED EXECUTION EXAMPLES")
    print("=" * 70)

    # Run local examples by default
    if not (args.remote or args.hybrid or args.all):
        await run_local_example()

    # Run requested examples
    if args.remote or args.all:
        await run_remote_example()

    if args.hybrid or args.all:
        await run_hybrid_example()

    if args.all:
        await run_local_example()

    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED:")
    print("=" * 70)
    print("""
1. @replica Decorator:
   - Marks classes for distributed execution
   - Can specify GPU requirements (gpus=N)
   - Supports backends (backend="grpc")
   - Handles stateful services

2. @node Decorator:
   - Marks pure functions for graph compilation
   - Used for data transformations
   - Automatically parallelized when possible

3. @jit Decorator:
   - Compiles entire pipelines into graphs
   - Enables optimization and parallelization
   - Works with both local and remote execution

4. Mesh:
   - Defines available computational resources
   - Can be local, remote, or hybrid
   - Provides context for execution (with mesh:)

5. Execution Patterns:
   - Local: All computation on local machine
   - Remote: Computation distributed to servers
   - Hybrid: Mix of local and remote resources

For more details, see:
- Core API: examples/demo.py
- gRPC serving: python -m thinkagain.serve --help
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

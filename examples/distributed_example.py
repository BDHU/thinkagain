"""Comprehensive distributed execution with dynamic Ray-style API.

This example demonstrates the new dynamic execution model:
1. @service decorator for mutable actor classes with .init()
2. @op decorator for pure functions with .go()
3. .go() calls return ObjectRef immediately (non-blocking)
4. Direct function composition: fn3.go(fn2.go(fn1.go(x)))
5. Mutable actor state (no decompose/compose needed)
6. Automatic background optimization
7. Mesh context for execution resources

Usage:
    # LOCAL EXECUTION (no servers needed):
    python examples/distributed_example.py

    # REMOTE EXECUTION (with gRPC servers):
    # Terminal 1 - Start LLM server:
    python -m thinkagain.serve examples.distributed_example:LLM --port 8000

    # Terminal 2 - Run client with remote mesh:
    python examples/distributed_example.py --remote

Key Changes from Old API:
- NO @jit decorator needed - graphs built dynamically at runtime
- NO @trace or decompose/compose - actors have mutable state
- .go() returns ObjectRef immediately (Ray-style)
- Actor methods: actor.method.go(x) instead of apply_replica()
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
# Define Services (Mutable Actors)
# ============================================================================


@ta.service()  # CPU-only service
class Retriever:
    """Document retriever - lightweight CPU actor.

    Demonstrates:
    - CPU-only workloads (no GPU requirement)
    - Lightweight initialization
    - Stateless operations
    """

    def __init__(self):
        print("  [INIT] Retriever initialized")

    async def retrieve(self, query: str) -> list[str]:
        """Retrieve documents for a query."""
        await asyncio.sleep(0.02)  # Simulate database query
        return [f"Document {i}: Information about '{query}'" for i in range(1, 4)]


@ta.service(gpus=1)  # GPU service (will run on CPU if no GPU available)
class LLM:
    """Mutable LLM actor with state management.

    Demonstrates:
    - GPU resource requirements (gpus=1, falls back to CPU if unavailable)
    - Heavy initialization (model loading)
    - Mutable state (temperature can be changed)
    - No decompose/compose needed
    """

    def __init__(self, model_name: str = "llama-70b", temperature: float = 0.7):
        """Initialize LLM.

        Args:
            model_name: Name of model to load
            temperature: Sampling temperature (can be updated)
        """
        self.model_name = model_name
        self.temperature = temperature
        # Heavy initialization (loads model into GPU memory)
        self.engine = MockEngine(model_name)

    async def generate(self, prompt: str) -> str:
        """Generate text from prompt using current temperature."""
        return await self.engine.generate(prompt, self.temperature)

    async def set_temperature(self, new_temp: float):
        """Update temperature (mutable state)."""
        print(f"  [UPDATE] LLM temperature: {self.temperature} -> {new_temp}")
        self.temperature = new_temp


@ta.service()
class Counter:
    """Simple counter demonstrating mutable state.

    Demonstrates:
    - Mutable state updates
    - Multiple method calls maintaining state
    """

    def __init__(self, start: int = 0):
        print(f"  [INIT] Counter initialized at {start}")
        self.count = start

    async def increment(self, amount: int) -> int:
        """Increment counter and return new value."""
        self.count += amount
        return self.count

    async def get(self) -> int:
        """Get current count."""
        return self.count


# ============================================================================
# Define Pure Functions (@op)
# ============================================================================


@ta.op
async def combine_docs(docs: list[str]) -> str:
    """Combine retrieved documents into a single context."""
    return "\\n\\n".join(docs)


@ta.op
async def format_response(response: str, query: str) -> dict:
    """Format LLM response into structured output."""
    return {
        "query": query,
        "response": response,
        "timestamp": "2026-01-13T00:00:00Z",
    }


# ============================================================================
# Pipelines (No @jit needed!)
# ============================================================================


async def rag_pipeline(retriever, llm, query: str) -> dict:
    """RAG (Retrieval-Augmented Generation) pipeline.

    Flow:
    1. Retrieve relevant documents
    2. Combine documents into context
    3. Generate response from LLM
    4. Format output

    Note: No @jit decorator needed! Dynamic execution builds the DAG
    automatically as .go() calls are made.
    """
    # Step 1: Retrieve documents (non-blocking)
    docs_ref = retriever.retrieve.go(query)

    # Step 2: Combine docs (non-blocking, depends on docs_ref)
    context_ref = combine_docs.go(docs_ref)

    # Step 3: Generate response (non-blocking, depends on context_ref)
    response_ref = llm.generate.go(context_ref)

    # Step 4: Format output (non-blocking, depends on response_ref)
    result_ref = format_response.go(response_ref, query)

    # Wait for final result (blocks only when we need the value)
    return await result_ref


async def counter_pipeline(counter) -> int:
    """Simple counter pipeline demonstrating mutable state."""
    # Increment multiple times - state persists!
    ref1 = counter.increment.go(10)
    result1 = await ref1
    print(f"  After increment(10): {result1}")

    ref2 = counter.increment.go(5)
    result2 = await ref2
    print(f"  After increment(5): {result2}")

    ref3 = counter.get.go()
    return await ref3


async def temperature_pipeline(llm) -> tuple[str, str]:
    """Pipeline demonstrating mutable actor state updates."""
    # Generate with initial temperature
    ref1 = llm.generate.go("What is machine learning?")
    response1 = await ref1

    # Update temperature (mutable state change)
    await llm.set_temperature.go(0.9)

    # Generate again with new temperature
    ref2 = llm.generate.go("What is machine learning?")
    response2 = await ref2

    return response1, response2


# ============================================================================
# Main Execution
# ============================================================================


async def run_local():
    """Run with local CPU-only execution."""
    print("=" * 70)
    print("LOCAL EXECUTION (CPU only)")
    print("=" * 70)

    # Create mesh with local CPU device
    mesh = ta.Mesh([ta.CpuDevice(0)])

    # Create actor handles BEFORE entering mesh context
    retriever = Retriever.init()  # type: ignore[attr-defined]
    llm = LLM.init(model_name="llama-7b", temperature=0.7)  # type: ignore[attr-defined]
    counter = Counter.init(start=0)  # type: ignore[attr-defined]

    print("\\nCreated actors:")
    print(f"  - Retriever: {retriever}")
    print(f"  - LLM: {llm}")
    print(f"  - Counter: {counter}")

    with mesh:
        print("\\n" + "=" * 70)
        print("Example 1: RAG Pipeline")
        print("=" * 70)
        result = await rag_pipeline(retriever, llm, "What is distributed computing?")
        print(f"\\nResult: {result}")

        print("\\n" + "=" * 70)
        print("Example 2: Counter Pipeline (Mutable State)")
        print("=" * 70)
        final_count = await counter_pipeline(counter)
        print(f"\\nFinal count: {final_count}")

        print("\\n" + "=" * 70)
        print("Example 3: Temperature Update (Mutable State)")
        print("=" * 70)
        response1, response2 = await temperature_pipeline(llm)
        print(f"\\nWith temp=0.7: {response1}")
        print(f"With temp=0.9: {response2}")


async def run_remote():
    """Run with remote gRPC execution."""
    print("=" * 70)
    print("REMOTE EXECUTION (gRPC)")
    print("=" * 70)
    print("Make sure LLM server is running:")
    print("  python -m thinkagain.serve examples.distributed_example:LLM --port 8000")
    print("=" * 70)

    # Create mesh with remote GPU device
    mesh = ta.Mesh(
        [
            ta.CpuDevice(0),  # Local CPU for lightweight work
            ta.MeshNode("gpu-server", gpus=1, endpoint="localhost:8000"),  # Remote GPU
        ]
    )

    # Create actor handles
    retriever = Retriever.init()  # type: ignore[attr-defined]
    llm = LLM.init(model_name="llama-70b", temperature=0.7)  # type: ignore[attr-defined]

    with mesh:
        print("\\nRunning RAG pipeline with remote LLM...")
        result = await rag_pipeline(retriever, llm, "What is quantum computing?")
        print(f"\\nResult: {result}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed execution example")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use remote gRPC execution (requires server running)",
    )
    args = parser.parse_args()

    if args.remote:
        await run_remote()
    else:
        await run_local()

    print("\\n" + "=" * 70)
    print("✅ Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

"""Comprehensive distributed execution and stateful replica patterns with thinkagain.

This example demonstrates:
1. @replica decorator for distributed classes
2. @node decorator for pure functions
3. @jit decorator for compiling pipelines
4. Mesh for defining execution resources
5. Stateful replica patterns with @trace
6. Local, remote, and hybrid execution

Usage:
    # LOCAL EXECUTION (no servers needed):
    python examples/distributed_example.py

    # REMOTE EXECUTION (with gRPC servers):
    # Terminal 1 - Start LLM server:
    python -m thinkagain.serve examples.distributed_example:LLM --port 8000

    # Terminal 2 - Run client with remote mesh:
    python examples/distributed_example.py --remote

Key Concepts:
- @replica: Marks stateful classes for distributed execution
- @trace: Enables pure functional updates via decompose/compose
- @node: Marks pure functions for graph compilation
- @jit: Compiles pipelines for optimization and distribution
- Mesh: Defines available computational resources
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

    Demonstrates:
    - CPU-only workloads (no GPU requirement)
    - Lightweight initialization
    - Stateless operations
    """

    def __init__(self):
        print("  [INIT] Retriever initialized")

    async def __call__(self, query: str) -> list[str]:
        """Retrieve documents for a query."""
        await asyncio.sleep(0.02)  # Simulate database query
        return [f"Document {i}: Information about '{query}'" for i in range(1, 4)]


@ta.trace
@ta.replica(gpus=1, backend="grpc")  # GPU replica with gRPC backend
class LLM:
    """Stateful LLM replica with @trace for pure updates.

    Demonstrates:
    - GPU resource requirements (gpus=1)
    - Heavy initialization (model loading)
    - Stateful operations with pure functional updates
    - Separating static state (aux) from dynamic state (children)
    """

    def __init__(self, model_name: str = "llama-70b", temperature: float = 0.7):
        """Initialize LLM.

        Args:
            model_name: Name of model to load (goes in aux - never changes)
            temperature: Sampling temperature (goes in children - can be updated)
        """
        self.model_name = model_name
        self.temperature = temperature
        # Heavy initialization (loads model into GPU memory)
        self.engine = MockEngine(model_name)
        self.cache = {}

    def decompose(self) -> tuple[list, dict]:
        """Decompose into runtime state (children) and static state (aux).

        Children: Values that can change via pure updates (temperature, cache)
        Aux: Static initialization data (model_name, loaded engine)
        """
        children = [self.temperature, self.cache]
        aux = {
            "model_name": self.model_name,
            "engine": self.engine,
        }
        return children, aux

    @classmethod
    def compose(cls, aux: dict, children: list) -> "LLM":
        """Reconstruct LLM with updated runtime state."""
        temperature, cache = children
        instance = cls.__new__(cls)
        instance.model_name = aux["model_name"]
        instance.engine = aux["engine"]
        instance.temperature = temperature
        instance.cache = cache
        return instance

    async def __call__(self, prompt: str) -> str:
        """Generate response (pure function of prompt + state)."""
        # Check cache
        if prompt in self.cache:
            return self.cache[prompt]

        # Generate response
        response = await self.engine.generate(prompt, self.temperature)
        return response


@ta.replica(backend="grpc")  # Simple text processing replica
class TextProcessor:
    """Stateful text processor for various transformations.

    Demonstrates:
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
            f"  [TextProcessor] Request #{self.request_count}: {operation}('{text}') -> '{result}'"
        )
        return result


@ta.trace
@ta.replica()
class Counter:
    """Simple counter replica with @trace for pure updates.

    Demonstrates:
    - Basic stateful replica with decompose/compose
    - Pure functional state management
    - Using apply_replica for updates
    """

    def __init__(self, count: int = 0, step: int = 1):
        """Initialize counter.

        Args:
            count: Initial count value
            step: Increment step size
        """
        self.count = count
        self.step = step

    def decompose(self) -> tuple[list, None]:
        """Break replica into traceable children."""
        return [self.count, self.step], None

    @classmethod
    def compose(cls, aux, children: list) -> "Counter":
        """Reconstruct replica from children."""
        count, step = children
        return cls(count=count, step=step)

    async def __call__(self, x: int) -> int:
        """Pure computation: count + step + x (doesn't mutate)."""
        return self.count + self.step + x


# ============================================================================
# Define Nodes (Pure Functions)
# ============================================================================


@ta.node
async def format_rag_prompt(query: str, docs: list[str]) -> str:
    """Format a RAG prompt from query and documents."""
    context = "\n".join(f"- {doc}" for doc in docs[:3])
    return f"""Context:
{context}

Query: {query}

Answer:"""


@ta.node
async def increment_counter_state(
    count: int, step: int, amount: int
) -> tuple[list, int]:
    """Pure function to update counter state.

    This demonstrates using pure functions to update replica state.

    Args:
        count: Current count (from decompose children[0])
        step: Current step (from decompose children[1])
        amount: Value to add

    Returns:
        (new_children, output) tuple
    """
    new_count = count + step + amount
    new_step = step
    return [new_count, new_step], new_count


@ta.node
async def update_temperature(
    temp: float, cache: dict, new_temp: float
) -> tuple[list, str]:
    """Pure function to update LLM temperature.

    Args:
        temp: Current temperature (from decompose children[0])
        cache: Current cache (from decompose children[1])
        new_temp: New temperature value

    Returns:
        (new_children, output) tuple
    """
    return [new_temp, cache], f"Temperature updated to {new_temp}"


# ============================================================================
# Create Replica Handles (Outside @jit)
# ============================================================================

# These handles are used within @jit pipelines
retriever = Retriever.init()  # type: ignore[attr-defined]
llm = LLM.init("llama-70b")  # type: ignore[attr-defined]
text_processor = TextProcessor.init()  # type: ignore[attr-defined]
counter = Counter.init(count=0, step=1)  # type: ignore[attr-defined]


# ============================================================================
# Define Pipelines with @jit
# ============================================================================


@ta.jit
async def rag_pipeline(query: str) -> str:
    """RAG (Retrieval-Augmented Generation) pipeline.

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
    response = await llm(prompt)

    return response


@ta.jit
async def text_processing_pipeline(text: str, operations: list[str]) -> dict:
    """Multi-operation text processing pipeline.

    Demonstrates parallel operations on the same input.
    """
    result = {"original": text}

    # Apply each operation
    for operation in operations:
        result[operation] = await text_processor(text, operation=operation)

    return result


@ta.jit
async def counter_pipeline(x: int) -> int:
    """Simple pipeline using counter replica."""
    result = await counter(x)
    return result


# ============================================================================
# Example Execution Functions
# ============================================================================


async def run_local_example():
    """Run examples with local mesh (no remote servers needed)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: LOCAL EXECUTION")
    print("=" * 70)
    print("\nRunning pipelines with local mesh...")

    # Create local mesh with 1 GPU
    mesh = ta.Mesh([ta.MeshNode("local", gpus=1)])

    with mesh:
        # Example 1a: Single RAG query
        print("\n[1a] RAG Pipeline:")
        result = await rag_pipeline("What is machine learning?")
        print("  Query: What is machine learning?")
        print(f"  Response: {result}")

        # Example 1b: Text processing pipeline
        print("\n[1b] Text Processing Pipeline:")
        text = "Hello Distributed World"
        operations = ["upper", "lower", "title", "reverse"]
        result = await text_processing_pipeline(text, operations)
        print(f"  Original: {result['original']}")
        for op in operations:
            print(f"  {op.capitalize()}: {result[op]}")

        # Example 1c: Counter with pure updates
        print("\n[1c] Counter with Pure State Updates:")
        print(f"  Initial: count={counter.count}, step={counter.step}")

        # Use counter in pipeline
        result = await counter_pipeline(10)
        print(f"  counter(10) = {result}")

        # Update counter state using apply_replica
        print("\n  Updating counter state with apply_replica...")
        out = await ta.apply_replica(counter, increment_counter_state, 100)
        print(
            f"  After increment_counter_state(100): count={counter.count}, output={out}"
        )


async def run_remote_example():
    """Run examples with remote mesh (requires gRPC servers)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: REMOTE EXECUTION")
    print("=" * 70)
    print("\nMake sure you've started the servers:")
    print(
        "  Terminal 1: python -m thinkagain.serve examples.distributed_example:LLM --port 8000"
    )
    print(
        "  Terminal 2: python -m thinkagain.serve examples.distributed_example:TextProcessor --port 8001"
    )

    # Create remote mesh with gRPC endpoints
    mesh = ta.Mesh(
        [
            ta.MeshNode("llm_server", endpoint="localhost:8000", gpus=1),
            ta.MeshNode("processor_server", endpoint="localhost:8001"),
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
            for op in operations:
                print(f"  {op.capitalize()}: {result[op]}")

    except Exception as e:
        print(f"\n❌ Error connecting to remote servers: {e}")
        print("\nMake sure servers are running with the commands above.")


async def run_stateful_patterns():
    """Demonstrate advanced stateful replica patterns."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: STATEFUL REPLICA PATTERNS")
    print("=" * 70)

    # Pattern 1: Basic replica with @trace
    print("\n[3a] Counter Replica with @trace:")
    local_counter = Counter(count=10, step=5)
    print(f"  Initial state: count={local_counter.count}, step={local_counter.step}")

    # Use apply_replica for pure updates
    out1 = await ta.apply_replica(local_counter, increment_counter_state, 100)
    print(f"  After increment(100): count={local_counter.count}, output={out1}")

    out2 = await ta.apply_replica(local_counter, increment_counter_state, 50)
    print(f"  After increment(50): count={local_counter.count}, output={out2}")

    # Pattern 2: LLM with runtime state
    print("\n[3b] LLM Replica with Runtime State:")
    llm_instance = LLM("gpt-4", temperature=0.5)
    print(f"  Initial temperature: {llm_instance.temperature}")

    # Update temperature with pure function
    msg = await ta.apply_replica(llm_instance, update_temperature, 0.9)
    print(f"  {msg}")
    print(f"  New temperature: {llm_instance.temperature}")


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ThinkAgain Distributed Execution Examples"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run remote execution examples (requires gRPC servers)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("THINKAGAIN DISTRIBUTED EXECUTION & STATEFUL PATTERNS")
    print("=" * 70)

    # Run local examples by default
    if not (args.remote or args.all):
        await run_local_example()
        await run_stateful_patterns()

    # Run requested examples
    if args.remote or args.all:
        await run_remote_example()

    if args.all:
        await run_local_example()
        await run_stateful_patterns()

    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED:")
    print("=" * 70)
    print("""
1. @replica Decorator:
   - Marks classes for distributed execution
   - Can specify GPU requirements (gpus=N)
   - Supports backends (backend="grpc")
   - Handles stateful services

2. @trace Decorator:
   - Enables pure functional updates via decompose/compose
   - Separates dynamic state (children) from static state (aux)
   - Works with apply_replica for pure state transformations

3. @node Decorator:
   - Marks pure functions for graph compilation
   - Used for data transformations
   - Automatically parallelized when possible

4. @jit Decorator:
   - Compiles entire pipelines into graphs
   - Enables optimization and parallelization
   - Works with both local and remote execution

5. Mesh:
   - Defines available computational resources
   - Can be local, remote, or hybrid
   - Provides context for execution (with mesh:)

6. Stateful Patterns:
   - decompose(): Break state into children + aux
   - compose(): Reconstruct from children + aux
   - apply_replica(): Pure functional updates

For more details, see:
- Core API: examples/demo.py
- Bundle API: examples/bundle_example.py
- Agent API: examples/agents/simple_agent.py
- gRPC serving: python -m thinkagain.serve --help
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

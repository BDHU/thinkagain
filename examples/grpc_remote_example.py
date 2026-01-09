"""Example demonstrating remote execution with gRPC backend.

This example shows the clean @replicate API:
1. Decorate a function/class with @replicate
2. Serve it directly with: python -m thinkagain.serve
3. Call it from @jit pipelines using a mesh

Usage:
    # Terminal 1 - Start server 1 (function):
    python -m thinkagain.serve examples.grpc_remote_example:reverse_text --port 8000

    # Terminal 2 - Start server 2 (function):
    python -m thinkagain.serve examples.grpc_remote_example:to_upper --port 8001

    # Terminal 3 - Start server 3 (function):
    python -m thinkagain.serve examples.grpc_remote_example:to_lower --port 8002

    # Terminal 4 (Optional) - Start server 4 (class):
    python -m thinkagain.serve examples.grpc_remote_example:TextProcessor --port 8003

    # Terminal 5 - Run client:
    python examples/grpc_remote_example.py

This demonstrates:
- Example 1: Function-based replicas in @jit pipelines
- Example 2: Class-based replicas in @jit pipelines (new feature!)
- Example 3: Direct gRPC client calls (low-level alternative)
"""

import asyncio

from thinkagain import jit
from thinkagain.distributed import Mesh, MeshNode, replicate


# ============================================================================
# Example 1: Function-based replication
# ============================================================================


# Just decorate your function with @replicate - that's it!
# The decorator makes it serveable AND callable in pipelines
@replicate(backend="grpc")
async def reverse_text(text: str) -> str:
    """Reverse a string.

    This function can be:
    1. Served with: python -m thinkagain.serve examples.grpc_remote_example:reverse_text
    2. Called from @jit pipelines with mesh context (routes to remote servers)
    3. Called directly for local testing

    Args:
        text: Input string

    Returns:
        Reversed string
    """
    # Implementation - this runs on the server
    result = text[::-1]
    print(f"[Server] Reversing: '{text}' -> '{result}'")
    return result


# ============================================================================
# Example 2: More function-based replicas with different operations
# ============================================================================


@replicate(backend="grpc")
async def to_upper(text: str) -> str:
    """Convert text to uppercase."""
    result = text.upper()
    print(f"[Server] to_upper: '{text}' -> '{result}'")
    return result


@replicate(backend="grpc")
async def to_lower(text: str) -> str:
    """Convert text to lowercase."""
    result = text.lower()
    print(f"[Server] to_lower: '{text}' -> '{result}'")
    return result


@replicate(backend="grpc")
async def to_title(text: str) -> str:
    """Convert text to title case."""
    result = text.title()
    print(f"[Server] to_title: '{text}' -> '{result}'")
    return result


# ============================================================================
# Example 3: Class-based replication with state
# ============================================================================


@replicate(backend="grpc")
class TextProcessor:
    """Stateful text processor that can be served remotely.

    This class can be:
    1. Served directly with: python -m thinkagain.serve examples.grpc_remote_example:TextProcessor
    2. Called from @jit pipelines (routes to mesh instances)
    3. Called directly from client code (as shown in Example 2)
    """

    def __init__(self):
        """Initialize the text processor with state."""
        self.request_count = 0
        print("[Server] TextProcessor initialized")

    async def __call__(self, text: str, operation: str = "upper") -> str:
        """Process text with various operations.

        Args:
            text: Input string
            operation: Operation to perform ("upper", "lower", "title", "swap")

        Returns:
            Processed string
        """
        self.request_count += 1

        if operation == "upper":
            result = text.upper()
        elif operation == "lower":
            result = text.lower()
        elif operation == "title":
            result = text.title()
        elif operation == "swap":
            result = text.swapcase()
        else:
            result = text

        print(
            f"[Server] Request #{self.request_count}: {operation}('{text}') -> '{result}'"
        )
        return result


# ============================================================================
# Client code that uses the remote replicas
# ============================================================================


async def main():
    """Run example client that calls remote replicas."""

    # Configure mesh with remote endpoints
    mesh = Mesh(
        [
            MeshNode("server1", endpoint="localhost:8000"),
            MeshNode("server2", endpoint="localhost:8001"),
            MeshNode("server3", endpoint="localhost:8002"),
        ]
    )

    print("=" * 60)
    print("Example 1: Complex pipeline with multiple replicas")
    print("=" * 60)

    # Define a real pipeline that composes multiple remote operations
    @jit
    async def text_processing_pipeline(text: str) -> dict:
        """Multi-step text processing pipeline using multiple remote replicas.

        This demonstrates the real power of @jit with @replicate:
        1. Reverse the text (remote function)
        2. Apply multiple transformations in parallel (remote functions)
        3. Combine results

        The @jit decorator allows for automatic parallelization and optimization.
        """
        # Step 1: Reverse the text using the remote function
        reversed_text = await reverse_text(text)

        # Step 2: Reverse it again for comparison (shows sequential dependency)
        double_reversed = await reverse_text(reversed_text)

        # Step 3: Apply different operations in parallel
        # These can potentially run in parallel since they don't depend on each other
        upper_result = await to_upper(reversed_text)
        lower_result = await to_lower(reversed_text)
        title_result = await to_title(reversed_text)

        # Step 4: Combine all results
        return {
            "original": text,
            "reversed": reversed_text,
            "double_reversed": double_reversed,
            "upper": upper_result,
            "lower": lower_result,
            "title": title_result,
        }

    # Execute the complex pipeline
    print("\nConnecting to remote servers...")
    print("Note: Make sure servers are running on ports 8000, 8001, 8002")
    print()
    with mesh:
        inputs = ["hello world", "distributed computing", "grpc rocks"]
        print(f"Processing {len(inputs)} inputs through pipeline...\n")

        for text in inputs:
            result = await text_processing_pipeline(text)
            print(f"Input: '{result['original']}'")
            print(f"  → Reversed: '{result['reversed']}'")
            print(f"  → Double Reversed: '{result['double_reversed']}'")
            print(f"  → Upper: '{result['upper']}'")
            print(f"  → Lower: '{result['lower']}'")
            print(f"  → Title: '{result['title']}'")
            print()

    print("=" * 60)
    print("Example 2: Class-based replica in @jit pipeline")
    print("=" * 60)
    print()
    print("This example shows that classes decorated with @replicate can now be")
    print("used inside @jit pipelines, just like functions!")
    print()

    # Configure mesh with TextProcessor endpoint
    mesh_with_class = Mesh(
        [
            MeshNode("processor", endpoint="localhost:8003"),
        ]
    )

    # Instantiate the processor
    processor = TextProcessor()

    @jit
    async def pipeline_with_class(text: str) -> dict:
        """Pipeline that uses a class-based replica.

        This demonstrates that @replicate now works with classes inside @jit!
        """
        upper = await processor(text, operation="upper")
        lower = await processor(text, operation="lower")
        title = await processor(text, operation="title")
        swap = await processor(text, operation="swap")

        return {
            "original": text,
            "upper": upper,
            "lower": lower,
            "title": title,
            "swap": swap,
        }

    print("Connecting to TextProcessor server on port 8003...")
    print(
        "Note: Start with: python -m thinkagain.serve examples.grpc_remote_example:TextProcessor --port 8003"
    )
    print()

    try:
        with mesh_with_class:
            test_text = "Hello World"
            result = await pipeline_with_class(test_text)

            print(f"Input: '{result['original']}'")
            print(f"  → Upper: '{result['upper']}'")
            print(f"  → Lower: '{result['lower']}'")
            print(f"  → Title: '{result['title']}'")
            print(f"  → Swap: '{result['swap']}'")
            print()
            print("✓ Class-based replica works inside @jit pipeline!")
    except Exception as e:
        print("Note: TextProcessor server not available on port 8003")
        print(f"Error: {e}")
        print("To test this example, start the server with:")
        print(
            "  python -m thinkagain.serve examples.grpc_remote_example:TextProcessor --port 8003"
        )

    print()
    print("=" * 60)
    print("Example 3: Direct class-based replica call (low-level)")
    print("=" * 60)
    print()
    print("Connecting to TextProcessor server on port 8003...")
    print(
        "Note: Start with: python -m thinkagain.serve examples.grpc_remote_example:TextProcessor --port 8003"
    )
    print()

    # Example 2: Call the class-based replica directly
    # Note: Classes can also be used inside @jit pipelines (like functions),
    # but this example shows direct gRPC client calls for demonstration
    from thinkagain.distributed.grpc.client import GrpcClient

    try:
        client = GrpcClient("localhost:8003")

        # Test various operations
        operations = [
            ("Hello World", "upper"),
            ("Hello World", "lower"),
            ("Hello World", "title"),
            ("Hello World", "swap"),
        ]

        for text, operation in operations:
            result = await client.execute(text, operation=operation)
            print(f"TextProcessor.{operation}('{text}') -> '{result}'")

        await client.close()
        print("\nTextProcessor example completed successfully!")

    except Exception as e:
        print("Note: TextProcessor server not available on port 8003")
        print(f"Error: {e}")
        print("To test this example, start the server with:")
        print(
            "  python -m thinkagain.serve examples.grpc_remote_example:TextProcessor --port 8003"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Clean @replicate API Example")
    print("=" * 60)
    print()
    print("Make sure you've started the server first:")
    print(
        "  python -m thinkagain.serve examples.grpc_remote_example:reverse_text --port 8000"
    )
    print()
    print("The @replicate decorator makes your function directly serveable!")
    print("No need to write separate server classes.")
    print()
    print("=" * 60)
    print()

    asyncio.run(main())

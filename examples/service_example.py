"""Example demonstrating the @replica decorator for stateful components."""

import asyncio

from thinkagain import Mesh, MeshNode, jit, node, replica


# ============================================================================
# Mock LLM (in real code, this would be vLLM, etc.)
# ============================================================================


class MockEngine:
    """Mock LLM engine for demonstration."""

    def __init__(self, model_name: str):
        print(f"  [INIT] Loading model: {model_name}")
        self.model_name = model_name

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        return f"[{self.model_name}@{temperature}]: Response to '{prompt}'"


# ============================================================================
# Stateful Replica
# ============================================================================


@replica(gpus=1)
class LLM:
    """Stateful LLM replica - heavy initialization, long-lived."""

    def __init__(self, model: str):
        # Heavy initialization (loads model)
        self.engine = MockEngine(model)

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        return await self.engine.generate(prompt, temperature)


# ============================================================================
# Stateless Node
# ============================================================================


@node
async def format_prompt(query: str) -> str:
    """Pure function - formats a query into a prompt."""
    return f"Query: {query}\nAnswer:"


# ============================================================================
# Create Replica Handle (Outside @jit)
# ============================================================================

llm = LLM.init("llama-70b")


# ============================================================================
# Pipeline
# ============================================================================


@jit
async def pipeline(query: str) -> str:
    """Stateless pipeline orchestrating replicas and nodes."""

    # Call stateless node
    prompt = await format_prompt(query)

    # Call stateful replica
    response = await llm.generate(prompt, temperature=0.7)

    return response


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run the replica example."""
    print("=" * 70)
    print("Replica Example")
    print("=" * 70)

    # Create mesh (local with 1 GPU)
    mesh = Mesh([MeshNode("local", gpus=1)])

    print("\nRunning pipeline with auto-deployed replica...")
    with mesh:
        # Replica auto-deploys on first call
        result = await pipeline("What is AI?")
        print(f"\nResult: {result}")

        # Second call reuses deployed replica
        result2 = await pipeline("What is Python?")
        print(f"\nResult: {result2}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

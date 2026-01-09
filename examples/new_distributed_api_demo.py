"""Demo of new distributed API with @replicate and Mesh.

This demonstrates the redesigned API where:
- @replicate marks functions for distributed execution
- Mesh defines available resources (explicit configuration)
- @jit compiles graphs (mesh-agnostic)
- Mesh is used as execution context (with mesh:)
- Optimization happens transparently (future implementation)
"""

import asyncio

import thinkagain as ta

# ============================================================================
# Define Mesh (Explicit Resource Configuration)
# ============================================================================

# Option 1: Auto-detect local GPUs
local_mesh = ta.Mesh(ta.devices())

# Option 2: Explicit multi-node cluster
cluster_mesh = ta.Mesh(
    [
        ta.MeshNode("server1", gpus=8, cpus=32, endpoint="server1:8000"),
        ta.MeshNode("server2", gpus=8, cpus=32, endpoint="server2:8000"),
    ]
)

print(f"Local mesh: {local_mesh}")
print(f"Cluster mesh: {cluster_mesh}")


# ============================================================================
# Define Replicated Functions
# ============================================================================


@ta.replicate  # CPU-only (no gpus parameter needed)
async def retrieve(query: str) -> list[str]:
    """Retrieve documents - can scale freely on CPU."""
    await asyncio.sleep(0.01)  # Simulate DB query
    return [f"doc_{i}: {query}" for i in range(3)]


@ta.replicate(gpus=4)  # 4 GPUs per instance
async def generate(query: str, docs: list[str]) -> str:
    """Generate response - requires GPU.

    Note: In the full implementation with setup parameter:

    def setup_llm():
        return LLM(device="cuda")

    @ta.replicate(gpus=4, setup=setup_llm)
    async def generate(llm, query: str, docs: list[str]) -> str:
        return await llm.generate(query, docs)
    """
    await asyncio.sleep(0.02)  # Simulate LLM call
    context = ", ".join(docs[:2])
    return f"Based on {context}: answer to '{query}'"


# ============================================================================
# Define Pipeline with @jit
# ============================================================================


@ta.jit  # Single decorator for local/distributed
async def rag_pipeline(query: str) -> str:
    """RAG pipeline - compiled once, runs on any mesh."""
    docs = await retrieve(query)
    response = await generate(query, docs)
    return response


# ============================================================================
# Execute Pipeline
# ============================================================================


async def main():
    print("\n" + "=" * 70)
    print("NEW DISTRIBUTED API DEMO")
    print("=" * 70)

    # Example 1: Run on local mesh
    print("\n[1] Running on local mesh (auto-detected GPUs)")
    with local_mesh:
        result = await rag_pipeline("What is machine learning?")
        print(f"Result: {result}")

    # Example 2: Run on cluster mesh
    print("\n[2] Running on cluster mesh (16 GPUs total)")
    with cluster_mesh:
        # In real implementation:
        # - First call: deploys with minimal instances (n=1 each)
        # - Profiles during execution
        # - Background optimizer scales up based on load
        # - Optimizer respects mesh.max_instances(gpus=4) = 4
        result = await rag_pipeline("Explain neural networks")
        print(f"Result: {result}")

    # Example 3: Multiple queries (would trigger auto-scaling in real impl)
    print("\n[3] Multiple queries (auto-scaling demonstration)")
    with cluster_mesh:
        queries = [
            "What is ML?",
            "Explain gradient descent",
            "What are transformers?",
        ]
        results = await asyncio.gather(*[rag_pipeline(q) for q in queries])
        for q, r in zip(queries, results):
            print(f"  Q: {q}")
            print(f"  A: {r[:60]}...")

    print("\n" + "=" * 70)
    print("Key Features:")
    print("- Single @jit decorator (no @pjit needed)")
    print("- Mesh is explicit (devices(), MeshNode)")
    print("- Mesh is execution context (with mesh:)")
    print("- @replicate for distributed functions")
    print("- Optimization is transparent (future)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

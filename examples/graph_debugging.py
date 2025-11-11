"""
Graph Debugging Example

Demonstrates how to use graph.steps() for:
- Interactive debugging
- State inspection
- Execution flow visualization
- Runtime state modification

Run with: python examples/graph_debugging.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Context, Worker, Graph, END


# ============================================================================
# Workers
# ============================================================================

class RetrieveWorker(Worker):
    def __call__(self, ctx: Context) -> Context:
        ctx.documents = ["Doc1", "Doc2"]
        ctx.log(f"[{self.name}] Retrieved {len(ctx.documents)} documents")
        return ctx


class CritiqueWorker(Worker):
    def __call__(self, ctx: Context) -> Context:
        # Quality assessment based on number of documents
        num_docs = len(ctx.documents)
        ctx.quality = 0.5 + (num_docs * 0.15)
        ctx.log(f"[{self.name}] Quality: {ctx.quality}")
        return ctx


class RefineWorker(Worker):
    def __call__(self, ctx: Context) -> Context:
        ctx.documents.append("Doc3")
        ctx.log(f"[{self.name}] Refined, now have {len(ctx.documents)} docs")
        return ctx


class GenerateWorker(Worker):
    def __call__(self, ctx: Context) -> Context:
        ctx.answer = f"Answer based on {len(ctx.documents)} documents"
        ctx.log(f"[{self.name}] Generated answer")
        return ctx


# ============================================================================
# Build Graph
# ============================================================================

def build_graph():
    """Build a simple graph with a cycle."""
    graph = Graph(name="debug_demo")

    graph.add_node("retrieve", RetrieveWorker())
    graph.add_node("critique", CritiqueWorker())
    graph.add_node("refine", RefineWorker())
    graph.add_node("generate", GenerateWorker())

    graph.set_entry("retrieve")

    graph.add_edge("retrieve", "critique")
    graph.add_conditional_edge(
        "critique",
        route=lambda ctx: "refine" if ctx.quality < 0.8 else "generate",
        paths={"refine": "refine", "generate": "generate"}
    )
    graph.add_edge("refine", "retrieve")  # Cycle
    graph.add_edge("generate", END)

    return graph


# ============================================================================
# Example 1: Basic Step-by-Step Debugging
# ============================================================================

def example_basic_steps():
    print("=" * 70)
    print("Example 1: Basic Step-by-Step Debugging")
    print("=" * 70)

    graph = build_graph()

    print("\n🔍 Step-by-Step Execution:\n")
    print("-" * 70)

    ctx = Context(query="What is ML?")

    for i, step in enumerate(graph.steps(ctx), 1):
        print(f"\n📍 Step {i}: {step.node}")
        print(f"   Next: {step.next_node}")

        # Inspect specific state
        if step.node == "retrieve":
            print(f"   Documents: {step.ctx.documents}")
        elif step.node == "critique":
            print(f"   Quality: {step.ctx.quality}")
        elif step.node == "refine":
            print(f"   Documents after refine: {step.ctx.documents}")
        elif step.node == "generate":
            print(f"   Answer: {step.ctx.answer}")

        print("-" * 70)

    print(f"\n✅ Final Answer: {ctx.answer}\n")


# ============================================================================
# Example 2: State Modification During Execution
# ============================================================================

def example_state_modification():
    print("=" * 70)
    print("Example 2: State Modification During Execution")
    print("=" * 70)

    graph = build_graph()

    print("\nWe'll inject a higher quality score to skip refinement...\n")
    print("-" * 70)

    ctx = Context(query="What is deep learning?")

    for i, step in enumerate(graph.steps(ctx), 1):
        print(f"\n📍 Step {i}: {step.node}")

        # Inject higher quality to test different path
        if step.node == "critique":
            original_quality = step.ctx.quality
            step.ctx.quality = 0.9  # Override!
            print(f"   🔧 Modified quality: {original_quality} → {step.ctx.quality}")
            print(f"   Next will be: {step.next_node}")

        print("-" * 70)

    print(f"\n✅ Final Answer: {ctx.answer}")
    print("\nNotice: We skipped the 'refine' step by modifying quality!\n")


# ============================================================================
# Main
# ============================================================================

def main():
    example_basic_steps()
    print("\n")
    example_state_modification()


if __name__ == "__main__":
    main()

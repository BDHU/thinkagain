"""
Graph Examples - Self-Correcting RAG Agent

Demonstrates Graph with cycles for complex workflows that require:
- Self-reflection and iterative refinement
- Conditional routing based on quality checks
- Loops back to previous nodes (unlike Pipeline)

Run with: python examples/graph_examples.py
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Context, Worker, Graph, END


# ============================================================================
# Workers
# ============================================================================

class RetrieveWorker(Worker):
    """Retrieves documents from a vector database."""

    def __call__(self, ctx: Context) -> Context:
        query = ctx.query

        # If we've refined the query, use the refined version
        if hasattr(ctx, 'refined_query'):
            query = ctx.refined_query
            ctx.log(f"[{self.name}] Using refined query: {query}")

        # Simulate improving retrieval on retries
        if not hasattr(ctx, 'retrieval_attempt') or ctx.retrieval_attempt is None:
            ctx.retrieval_attempt = 0
        ctx.retrieval_attempt += 1

        num_docs = min(2 + ctx.retrieval_attempt, 5)
        ctx.documents = [f"Document {i} about {query}" for i in range(num_docs)]

        ctx.log(f"[{self.name}] Retrieved {len(ctx.documents)} documents")
        return ctx


class GenerateWorker(Worker):
    """Generates an answer from retrieved documents."""

    def __call__(self, ctx: Context) -> Context:
        docs_summary = f"{len(ctx.documents)} documents"
        ctx.answer = f"Based on {docs_summary}: [Generated answer for '{ctx.query}']"
        ctx.log(f"[{self.name}] Generated answer")
        return ctx


class CritiqueWorker(Worker):
    """Critiques the generated answer for quality."""

    def __call__(self, ctx: Context) -> Context:
        # Quality improves with more documents
        num_docs = len(ctx.documents)

        if num_docs >= 4:
            ctx.quality_score = 0.9
            ctx.quality_issues = []
        elif num_docs >= 3:
            ctx.quality_score = 0.7
            ctx.quality_issues = ["Could be more comprehensive"]
        else:
            ctx.quality_score = 0.5
            ctx.quality_issues = ["Insufficient evidence", "Needs more sources"]

        ctx.log(f"[{self.name}] Quality score: {ctx.quality_score}")
        if ctx.quality_issues:
            ctx.log(f"[{self.name}] Issues: {', '.join(ctx.quality_issues)}")

        return ctx


class RefineWorker(Worker):
    """Refines the query to improve retrieval."""

    def __call__(self, ctx: Context) -> Context:
        original = ctx.query
        ctx.refined_query = f"{original} (detailed, comprehensive)"
        ctx.log(f"[{self.name}] Refined query based on critique")
        return ctx


# ============================================================================
# Build Graph with Cycle
# ============================================================================

def build_self_correcting_rag() -> Graph:
    """
    Build a self-correcting RAG agent using Graph with cycles.

    Flow:
        retrieve → generate → critique
                       ↑           ↓
                       ←── refine ←

    The cycle allows the agent to refine and retry until quality is acceptable.
    """
    graph = Graph(name="self_correcting_rag")

    # Add nodes
    graph.add_node("retrieve", RetrieveWorker())
    graph.add_node("generate", GenerateWorker())
    graph.add_node("critique", CritiqueWorker())
    graph.add_node("refine", RefineWorker())

    # Set entry point
    graph.set_entry("retrieve")

    # Define edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")

    # Conditional routing from critique
    def route_after_critique(ctx: Context) -> str:
        """Route based on quality score."""
        if ctx.quality_score >= 0.8:
            return "done"  # High quality, terminate
        elif ctx.retrieval_attempt >= 3:
            return "done"  # Max attempts reached
        else:
            return "refine"  # Low quality, try to improve

    graph.add_conditional_edge(
        "critique",
        route=route_after_critique,
        paths={
            "done": END,
            "refine": "refine"
        }
    )

    # The CYCLE: refine loops back to retrieve
    graph.add_edge("refine", "retrieve")

    return graph


# ============================================================================
# Example Usage
# ============================================================================

def main():
    print("=" * 70)
    print("Self-Correcting RAG Agent with Cycles")
    print("=" * 70)

    # Build the graph
    agent = build_self_correcting_rag()

    # Visualize the graph structure
    print("\n📊 Graph Structure (Mermaid):")
    print("-" * 70)
    print(agent.visualize())
    print("-" * 70)

    # Export graph structure as dict
    print("\n📋 Graph Structure (Dict):")
    print("-" * 70)
    print(json.dumps(agent.to_dict(), indent=2))
    print("-" * 70)

    # Execute the graph
    print("\n🚀 Executing Agent:")
    print("-" * 70)

    ctx = Context(query="What is machine learning?")
    result = agent.run(ctx)

    print("\n" + "=" * 70)
    print("✅ Execution Results")
    print("=" * 70)

    print(f"\n📝 Final Answer:")
    print(result.answer)

    print(f"\n📈 Quality Score: {result.quality_score}")
    print(f"🔄 Retrieval Attempts: {result.retrieval_attempt}")
    print(f"🛤️  Execution Path: {' → '.join(result.execution_path)}")
    print(f"📊 Total Steps: {result.total_steps}")

    print(f"\n📜 Execution History:")
    print("-" * 70)
    for entry in result.history:
        print(f"  {entry}")

    print("\n" + "=" * 70)
    print("🎯 Key Takeaway")
    print("=" * 70)
    print("The agent automatically cycled back to retrieval when quality was low,")
    print("demonstrating true cycle support that Pipeline cannot achieve!")
    print("=" * 70)


if __name__ == "__main__":
    main()

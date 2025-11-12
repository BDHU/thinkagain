"""
Subgraph Composition Example - Multi-Agent System

Demonstrates how to compose graphs together naturally.
Shows that graphs, pipelines, and workers are all just executables
that can be nested arbitrarily.

This example builds a multi-agent content creation system where:
- Each agent is its own graph (subgraph)
- A coordinator graph orchestrates the agents
- Agents can be reused across different coordinator graphs

Run with: python examples/subgraph_composition.py
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Context, Worker, Graph, END


# ============================================================================
# Workers for Research Agent
# ============================================================================

class QueryAnalyzer(Worker):
    """Analyzes and refines research queries."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Analyzing query: {ctx.query}")
        ctx.refined_query = f"{ctx.query} (comprehensive research)"
        ctx.search_terms = [ctx.query, "background", "recent developments"]
        return ctx


class WebSearcher(Worker):
    """Simulates web search."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Searching for: {ctx.refined_query}")
        # Simulate finding sources
        ctx.sources = [
            {"title": "Source 1", "content": f"Content about {ctx.query}"},
            {"title": "Source 2", "content": f"More about {ctx.query}"},
            {"title": "Source 3", "content": f"Research on {ctx.query}"},
        ]
        ctx.log(f"[{self.name}] Found {len(ctx.sources)} sources")
        return ctx


class SourceValidator(Worker):
    """Validates source quality."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Validating {len(ctx.sources)} sources")
        # Simulate validation
        ctx.validated_sources = ctx.sources[:2]  # Keep top 2
        ctx.source_quality = 0.85
        ctx.log(f"[{self.name}] Quality score: {ctx.source_quality}")
        return ctx


# ============================================================================
# Workers for Writing Agent
# ============================================================================

class OutlineGenerator(Worker):
    """Generates content outline."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Creating outline")
        ctx.outline = [
            "Introduction",
            "Key Concepts",
            "Applications",
            "Conclusion"
        ]
        ctx.log(f"[{self.name}] Generated {len(ctx.outline)} sections")
        return ctx


class ContentWriter(Worker):
    """Writes content based on outline."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Writing content")
        # Simulate writing
        ctx.draft = f"# {ctx.query}\n\n"
        for section in ctx.outline:
            ctx.draft += f"## {section}\n\nContent for {section}...\n\n"

        source_count = len(ctx.validated_sources) if hasattr(ctx, 'validated_sources') else 0
        ctx.draft += f"\n*Based on {source_count} sources*"

        ctx.log(f"[{self.name}] Draft length: {len(ctx.draft)} chars")
        return ctx


# ============================================================================
# Workers for Editing Agent
# ============================================================================

class GrammarChecker(Worker):
    """Checks grammar and style."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Checking grammar")
        ctx.grammar_score = 0.92
        ctx.grammar_issues = ["Minor issue 1", "Minor issue 2"]
        ctx.log(f"[{self.name}] Grammar score: {ctx.grammar_score}")
        return ctx


class ContentPolisher(Worker):
    """Polishes final content."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Polishing content")
        # Simulate polishing
        ctx.final_content = ctx.draft.replace("...", " [polished]")
        ctx.quality_score = (ctx.source_quality + ctx.grammar_score) / 2
        ctx.log(f"[{self.name}] Final quality: {ctx.quality_score:.2f}")
        return ctx


# ============================================================================
# Build Subgraphs (Agents)
# ============================================================================

def build_research_agent() -> Graph:
    """
    Build research agent as a reusable subgraph.

    Flow: analyze → search → validate
    """
    agent = Graph(name="research_agent")

    agent.add_node("analyze", QueryAnalyzer())
    agent.add_node("search", WebSearcher())
    agent.add_node("validate", SourceValidator())

    agent.set_entry("analyze")
    agent.add_edge("analyze", "search")
    agent.add_edge("search", "validate")
    agent.add_edge("validate", END)

    return agent


def build_writing_agent() -> Graph:
    """
    Build writing agent as a reusable subgraph.

    Flow: outline → write
    """
    agent = Graph(name="writing_agent")

    agent.add_node("outline", OutlineGenerator())
    agent.add_node("write", ContentWriter())

    agent.set_entry("outline")
    agent.add_edge("outline", "write")
    agent.add_edge("write", END)

    return agent


def build_editing_agent() -> Graph:
    """
    Build editing agent as a reusable subgraph.

    Flow: grammar_check → polish
    """
    agent = Graph(name="editing_agent")

    agent.add_node("grammar_check", GrammarChecker())
    agent.add_node("polish", ContentPolisher())

    agent.set_entry("grammar_check")
    agent.add_edge("grammar_check", "polish")
    agent.add_edge("polish", END)

    return agent


# ============================================================================
# Build Coordinator Graph with Subgraphs
# ============================================================================

def build_content_creation_system() -> Graph:
    """
    Build coordinator graph that orchestrates agent subgraphs.

    This demonstrates the key insight:
    - Each agent is a Graph
    - The coordinator is also a Graph
    - Graphs compose naturally as nodes

    Flow: research → write → edit
    """
    # Create the agent subgraphs
    research_agent = build_research_agent()
    writing_agent = build_writing_agent()
    editing_agent = build_editing_agent()

    # Create coordinator graph
    coordinator = Graph(name="content_creation_coordinator")

    # Add subgraphs as nodes!
    # This is the magic - graphs are just executables
    coordinator.add_node("research", research_agent)
    coordinator.add_node("write", writing_agent)
    coordinator.add_node("edit", editing_agent)

    # Wire them together
    coordinator.set_entry("research")
    coordinator.add_edge("research", "write")
    coordinator.add_edge("write", "edit")
    coordinator.add_edge("edit", END)

    return coordinator


# ============================================================================
# Alternative: Using >> Operator for Composition
# ============================================================================

def build_content_creation_pipeline() -> Graph:
    """
    Alternative approach: use >> operator for linear composition.

    This creates the same structure but more concisely.
    Under the hood, this is still a Graph!
    """
    research = build_research_agent()
    writing = build_writing_agent()
    editing = build_editing_agent()

    # Compose subgraphs with >> operator
    # Everything is composable!
    return research >> writing >> editing


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    print("=" * 80)
    print("Multi-Agent Content Creation System")
    print("Demonstrates Subgraph Composition")
    print("=" * 80)

    # Build the system
    system = build_content_creation_system()

    # Visualize the structure
    print("\n📊 System Architecture (Mermaid):")
    print("-" * 80)
    print(system.visualize())
    print("-" * 80)

    # Show structure
    print("\n🏗️  System Structure:")
    print(f"  Coordinator: {system.name}")
    print(f"  Total nodes: {len(system.nodes)}")
    print(f"  Subgraphs:")
    for node_name, node in system.nodes.items():
        if isinstance(node, Graph):
            print(f"    - {node_name}: {node.name} ({len(node.nodes)} internal nodes)")

    # Execute the system
    print("\n🚀 Executing Content Creation:")
    print("-" * 80)

    ctx = Context(query="Artificial Intelligence in Healthcare")
    result = await system.arun(ctx)

    print("\n" + "=" * 80)
    print("✅ Execution Results")
    print("=" * 80)

    print(f"\n📝 Final Content Preview:")
    print("-" * 80)
    preview = result.final_content[:300] + "..." if len(result.final_content) > 300 else result.final_content
    print(preview)
    print("-" * 80)

    print(f"\n📊 Quality Metrics:")
    print(f"  - Sources found: {len(result.sources)}")
    print(f"  - Sources validated: {len(result.validated_sources)}")
    print(f"  - Source quality: {result.source_quality:.2%}")
    print(f"  - Grammar score: {result.grammar_score:.2%}")
    print(f"  - Overall quality: {result.quality_score:.2%}")

    print(f"\n🛤️  Execution Path:")
    print(f"  {' → '.join(result.execution_path)}")

    print(f"\n📜 Execution Log:")
    print("-" * 80)
    for entry in result.history[-10:]:  # Show last 10 log entries
        print(f"  {entry}")

    print("\n" + "=" * 80)
    print("🎯 Key Insights")
    print("=" * 80)
    print("1. Each agent is a Graph (research_agent, writing_agent, editing_agent)")
    print("2. The coordinator is also a Graph that contains other Graphs as nodes")
    print("3. Subgraphs are added with simple add_node() - no special API")
    print("4. Context flows naturally through all levels")
    print("5. Everything composes with >> operator")
    print("\n💡 This is the power of \"everything is a Graph\"")
    print("=" * 80)

    # Show alternative composition
    print("\n\n" + "=" * 80)
    print("Alternative: Pipeline Composition")
    print("=" * 80)

    pipeline_system = build_content_creation_pipeline()
    print(f"\nUsing >> operator creates the same structure:")
    print(f"  {pipeline_system.name}")
    print(f"  Nodes: {len(pipeline_system.nodes)}")
    print(f"\nBoth approaches are equivalent - choose what's clearer!")


if __name__ == "__main__":
    asyncio.run(main())

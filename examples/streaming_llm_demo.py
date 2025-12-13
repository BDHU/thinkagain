"""
Example: Streaming LLM Worker
==============================

This example demonstrates how to create a streaming worker that yields
incremental updates during execution (e.g., token-by-token LLM streaming).
"""

import asyncio
from thinkagain import Context, Worker, Graph, END


class MockStreamingLLM(Worker):
    """
    Example streaming worker that simulates token-by-token LLM output.

    In a real implementation, this would call an actual LLM API that supports
    streaming (like OpenAI, Anthropic, etc.).
    """

    async def astream(self, ctx: Context):
        """Stream response tokens incrementally."""
        # Simulate streaming tokens from an LLM
        words = ["Hello", "world!", "This", "is", "a", "streaming", "response."]
        chunks = []

        for word in words:
            # Simulate network delay
            await asyncio.sleep(0.2)

            chunks.append(word)
            # Update context with accumulated response
            ctx.response = " ".join(chunks)

            # Yield context snapshot with current progress
            yield ctx

        # Final yield contains complete response
        # (already in ctx.response from last iteration)


class ProcessQuery(Worker):
    """Regular non-streaming worker."""

    async def arun(self, ctx: Context) -> Context:
        ctx.processed_query = ctx.user_query.strip().lower()
        ctx.log(f"[{self.name}] Processed query: {ctx.processed_query}")
        return ctx


async def demo_streaming_in_graph():
    """Demonstrate streaming worker in a graph."""
    print("=" * 60)
    print("Demo: Streaming Worker in Graph")
    print("=" * 60)

    # Build graph with streaming LLM
    graph = Graph(name="streaming_demo")
    graph.add_node("process", ProcessQuery())
    graph.add_node("llm", MockStreamingLLM())

    graph.set_entry("process")
    graph.add_edge("process", "llm")
    graph.add_edge("llm", END)

    # Create context
    ctx = Context(user_query="Hello, how are you?")

    print("\nStreaming execution:")
    print("-" * 60)

    # Stream execution - see incremental updates
    async for event in graph.stream(ctx):
        if event.type == "node" and event.streaming:
            # This is an intermediate update
            if hasattr(event.ctx, "response"):
                print(f"[Streaming] {event.ctx.response}")
        elif event.type == "node" and not event.streaming:
            # This is the final result for the node
            print(f"[Completed] {event.node}")

    print("\nFinal response:", ctx.response)
    print("=" * 60)


async def demo_non_streaming_mode():
    """Demonstrate that streaming workers also work in non-streaming mode."""
    print("\n" + "=" * 60)
    print("Demo: Same Worker in Non-Streaming Mode")
    print("=" * 60)

    # Same graph as before
    graph = Graph(name="non_streaming_demo")
    graph.add_node("llm", MockStreamingLLM())
    graph.add_edge("llm", END)

    ctx = Context(user_query="Hello!")

    # Execute without streaming - just get final result
    result = await graph.arun(ctx)

    print("\nFinal response:", result.response)
    print("=" * 60)


async def demo_streaming_worker_directly():
    """Demonstrate calling streaming worker directly."""
    print("\n" + "=" * 60)
    print("Demo: Call Streaming Worker Directly")
    print("=" * 60)

    worker = MockStreamingLLM()
    ctx = Context(user_query="Test")

    print("\nStreaming output:")
    async for ctx_snapshot in worker.astream(ctx):
        print(f"  -> {ctx_snapshot.response}")

    print("=" * 60)


if __name__ == "__main__":
    print("\nThinkAgain Streaming Demo")
    print("=" * 60)

    # Run all demos
    asyncio.run(demo_streaming_in_graph())
    asyncio.run(demo_non_streaming_mode())
    asyncio.run(demo_streaming_worker_directly())

    print("\nAll demos completed!")

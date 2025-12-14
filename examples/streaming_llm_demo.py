"""
Example: Streaming LLM Executable
=================================

Demonstrates how to create a streaming executable that yields
incremental updates during execution (e.g., token-by-token LLM streaming).
"""

import asyncio
from thinkagain import Context, Executable, Graph, END


class MockStreamingLLM(Executable):
    """
    Streaming executable that simulates token-by-token LLM output.

    In a real implementation, this would call an actual LLM API
    (OpenAI, Anthropic, etc.).
    """

    async def astream(self, ctx: Context):
        """Stream response tokens incrementally."""
        words = ["Hello", "world!", "This", "is", "a", "streaming", "response."]
        chunks = []

        for word in words:
            await asyncio.sleep(0.2)
            chunks.append(word)
            ctx.response = " ".join(chunks)
            yield ctx


class ProcessQuery(Executable):
    """Regular non-streaming executable."""

    async def arun(self, ctx: Context) -> Context:
        ctx.processed_query = ctx.user_query.strip().lower()
        ctx.log(f"[{self.name}] Processed query: {ctx.processed_query}")
        return ctx


async def demo_streaming_in_graph():
    print("=" * 60)
    print("Demo: Streaming Executable in Graph")
    print("=" * 60)

    graph = Graph(name="streaming_demo")
    graph.add("process", ProcessQuery())
    graph.add("llm", MockStreamingLLM())
    graph.edge("process", "llm")
    graph.edge("llm", END)

    ctx = Context(user_query="Hello, how are you?")

    print("\nStreaming execution:")
    print("-" * 60)

    compiled = graph.compile()
    async for event in compiled.stream(ctx):
        if event.type == "node" and event.streaming:
            if hasattr(event.ctx, "response"):
                print(f"[Streaming] {event.ctx.response}")
        elif event.type == "node" and not event.streaming:
            print(f"[Completed] {event.node}")

    print("=" * 60)


async def demo_non_streaming_mode():
    print("\n" + "=" * 60)
    print("Demo: Same Executable in Non-Streaming Mode")
    print("=" * 60)

    graph = Graph(name="non_streaming_demo")
    graph.add("llm", MockStreamingLLM())
    graph.edge("llm", END)

    ctx = Context(user_query="Hello!")
    result = await graph.compile().arun(ctx)

    print("\nFinal response:", result.response)
    print("=" * 60)


async def demo_streaming_directly():
    print("\n" + "=" * 60)
    print("Demo: Call Streaming Executable Directly")
    print("=" * 60)

    executable = MockStreamingLLM()
    ctx = Context(user_query="Test")

    print("\nStreaming output:")
    async for ctx_snapshot in executable.astream(ctx):
        print(f"  -> {ctx_snapshot.response}")

    print("=" * 60)


if __name__ == "__main__":
    print("\nThinkAgain Streaming Demo")
    print("=" * 60)

    asyncio.run(demo_streaming_in_graph())
    asyncio.run(demo_non_streaming_mode())
    asyncio.run(demo_streaming_directly())

    print("\nAll demos completed!")

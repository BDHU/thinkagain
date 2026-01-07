"""Simple agent example using the high-level Agent API.

This example demonstrates the easiest way to use ThinkAgain agents:
- Simple Agent class instantiation
- One-shot execution with .run()
- Multi-turn conversations by reusing AgentState
- Automatic tool calling
"""

import asyncio

from thinkagain.agents import Agent, tool, VLLM


@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A Python expression to evaluate (e.g., "15 * 234")

    Returns:
        The result of the calculation
    """
    try:
        # Safe eval with no builtins
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
async def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city to get weather for
        units: Temperature units - either 'celsius' or 'fahrenheit'

    Returns:
        Current weather information including temperature and conditions
    """
    temp = 72 if units == "fahrenheit" else 22
    return f"Weather in {city}: Sunny, {temp}°{'F' if units == 'fahrenheit' else 'C'}"


async def example_one_shot():
    """Example: One-shot execution with Agent.run()"""
    print("=" * 60)
    print("Example 1: One-shot execution")
    print("=" * 60)

    # Create agent with vLLM provider
    agent = Agent(
        model=VLLM(
            base_url="http://localhost:8000/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        tools=[calculate, get_weather],
        max_iterations=10,
    )

    # Run a single query
    state = await agent.run("What is 15 * 234? Also, what's the weather in Paris?")

    # Access the response
    print(f"\nFinal response: {state.messages[-1].content}")
    print(f"Iterations: {state.metadata['iterations']}")
    print(f"Stop reason: {state.metadata['stop_reason']}")
    print(f"Total messages: {len(state.messages)}")


async def example_multi_turn():
    """Example: Multi-turn conversation by reusing AgentState"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-turn conversation")
    print("=" * 60)

    # Create agent with system prompt
    agent = Agent(
        model=VLLM(
            base_url="http://localhost:8000/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        tools=[calculate, get_weather],
        system_prompt="You are a helpful assistant. Be concise and friendly.",
    )

    # Send multiple messages
    print("\nUser: What is 100 * 50?")
    state = await agent.run("What is 100 * 50?")
    print(f"Assistant: {state.messages[-1].content}")

    print("\nUser: What's the weather in Tokyo?")
    state = await agent.run("What's the weather in Tokyo?", state=state)
    print(f"Assistant: {state.messages[-1].content}")

    print("\nUser: What was my first question?")
    state = await agent.run("What was my first question?", state=state)
    print(f"Assistant: {state.messages[-1].content}")

    # Show full conversation
    print("\n" + "-" * 60)
    print("Full conversation history:")
    print("-" * 60)
    for msg in state.messages:
        print(f"{msg.role.capitalize()}: {msg.content}")


async def example_simple_math():
    """Example: Simplest possible usage"""
    print("\n" + "=" * 60)
    print("Example 3: Minimal usage (3 lines of code)")
    print("=" * 60)

    agent = Agent(
        model=VLLM(
            base_url="http://localhost:8000/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        tools=[calculate],
    )

    state = await agent.run("Calculate 789 * 456")
    print(f"\n{state.messages[-1].content}")


async def example_no_tools():
    """Example: Agent without tools (pure chat)"""
    print("\n" + "=" * 60)
    print("Example 4: Agent without tools (pure LLM)")
    print("=" * 60)

    agent = Agent(
        model=VLLM(
            base_url="http://localhost:8000/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        system_prompt="You are a helpful assistant. Answer questions concisely.",
    )

    state = await agent.run("What is the capital of France?")
    print(f"\n{state.messages[-1].content}")


async def main():
    """Run all examples."""
    # Run examples
    await example_one_shot()
    await example_multi_turn()
    await example_simple_math()
    await example_no_tools()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

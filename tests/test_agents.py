"""Tests for agent framework."""

import pytest

from thinkagain.agents import AgentState, Message, ToolCall


class MockProvider:
    def __init__(self, response):
        self._response = response

    async def complete(self, messages, tools=None, **kwargs):
        return self._response


def test_message_creation():
    """Test creating messages."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_message_with_tool_calls():
    """Test message with tool calls."""
    tool_call = ToolCall(id="call_123", name="search", arguments={"query": "python"})
    msg = Message(role="assistant", content="Let me search", tool_calls=[tool_call])
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "search"


def test_agent_state_creation():
    """Test creating agent state."""
    state = AgentState()
    assert state.messages == []

    state = AgentState(messages=[Message(role="user", content="Hi")])
    assert len(state.messages) == 1


def test_agent_state_immutability():
    """Test that agent state can be updated immutably."""
    state1 = AgentState(messages=[Message(role="user", content="Hi")])
    state2 = AgentState(
        messages=state1.messages + [Message(role="assistant", content="Hello")]
    )

    assert len(state1.messages) == 1
    assert len(state2.messages) == 2
    assert state1.messages[0].content == "Hi"
    assert state2.messages[1].content == "Hello"


@pytest.mark.asyncio
async def test_function_to_schema():
    """Test function to schema conversion."""
    from thinkagain.agents.llm import _function_to_schema

    async def search(query: str, limit: int = 10) -> str:
        """Search for information."""
        return f"Results for {query}"

    schema = _function_to_schema(search)

    assert schema["name"] == "search"
    assert schema["description"] == "Search for information."
    assert "query" in schema["parameters"]["properties"]
    assert "limit" in schema["parameters"]["properties"]
    assert "query" in schema["parameters"]["required"]
    assert "limit" not in schema["parameters"]["required"]  # Has default


@pytest.mark.asyncio
async def test_llm_node_basic():
    """Test basic LLM node execution."""
    from thinkagain import Context
    from thinkagain.agents import call_llm

    provider = MockProvider(
        {
            "role": "assistant",
            "content": "Hello! I'm a mock assistant.",
        }
    )
    state = AgentState(messages=[Message(role="user", content="Hi")])

    ctx = call_llm(Context(state), provider=provider)
    result = await ctx

    assert len(result.messages) == 2
    assert result.messages[1].role == "assistant"
    assert result.messages[1].content == "Hello! I'm a mock assistant."


@pytest.mark.asyncio
async def test_llm_node_with_tools():
    """Test LLM node with tool calls."""
    from thinkagain import Context
    from thinkagain.agents import call_llm

    async def search(query: str) -> str:
        """Search tool."""
        return f"Results for {query}"

    provider = MockProvider(
        {
            "role": "assistant",
            "content": "Let me search for that",
            "tool_calls": [
                {
                    "id": "call_123",
                    "name": "search",
                    "arguments": {"query": "python"},
                }
            ],
        }
    )
    state = AgentState(messages=[Message(role="user", content="Search for python")])
    tools = [search]

    ctx = call_llm(Context(state), provider=provider, tools=tools)
    result = await ctx

    assert len(result.messages) == 2
    assert result.messages[1].tool_calls is not None
    assert len(result.messages[1].tool_calls) == 1
    assert result.messages[1].tool_calls[0].name == "search"


@pytest.mark.asyncio
async def test_execute_tools_node():
    """Test tool execution node."""
    from thinkagain.agents import execute_tools

    async def calculate(expression: str) -> int:
        """Calculate expression."""
        return eval(expression)

    tools = [calculate]

    # State with tool call in last message
    state = AgentState(
        messages=[
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content="Let me calculate",
                tool_calls=[
                    ToolCall(
                        id="call_123", name="calculate", arguments={"expression": "2+2"}
                    )
                ],
            ),
        ]
    )

    result = await execute_tools(state, tools=tools)

    # Should have added tool result message
    assert len(result.messages) == 3
    assert result.messages[2].role == "tool"
    assert result.messages[2].content == "4"
    assert result.messages[2].tool_call_id == "call_123"


@pytest.mark.asyncio
async def test_execute_tools_no_calls():
    """Test tool execution with no tool calls."""
    from thinkagain.agents import execute_tools

    tools = []
    state = AgentState(
        messages=[
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello"),
        ]
    )

    result = await execute_tools(state, tools=tools)

    # Should return unchanged state
    assert len(result.messages) == 2


@pytest.mark.asyncio
async def test_execute_tool_call_node():
    """Test single tool call execution node."""
    from thinkagain import Context
    from thinkagain.agents.llm import execute_tool_call

    async def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    state = AgentState(messages=[Message(role="user", content="Calculate")])
    tool_call = ToolCall(id="call_456", name="multiply", arguments={"a": 7, "b": 6})

    ctx = execute_tool_call(Context(state), tool_call=tool_call, tool_fn=multiply)
    result = await ctx

    # Should have added tool result message
    assert len(result.messages) == 2
    assert result.messages[1].role == "tool"
    assert result.messages[1].content == "42"
    assert result.messages[1].tool_call_id == "call_456"


@pytest.mark.asyncio
async def test_agent_basic():
    """Test basic Agent class usage."""
    from thinkagain.agents import Agent

    provider = MockProvider({"role": "assistant", "content": "Hello!"})
    agent = Agent(model=provider, max_iterations=1)
    result = await agent.run("Hi")

    assert result.messages[-1].content == "Hello!"
    assert result.metadata["stop_reason"] == "complete"


@pytest.mark.asyncio
async def test_agent_state_reuse():
    """Test multi-turn usage by reusing AgentState."""
    from thinkagain.agents import Agent

    provider = MockProvider({"role": "assistant", "content": "Response"})
    agent = Agent(model=provider, max_iterations=1)

    state = await agent.run("First")
    state = await agent.run("Second", state=state)

    assert len(state.messages) == 4  # 2 user + 2 assistant

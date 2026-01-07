"""High-level Agent class for easy agent creation."""

from __future__ import annotations

from typing import Any, Callable

from thinkagain import Context

from .llm import LLM, call_llm, execute_tools
from .providers import OpenAI
from .state import AgentState, Message


class Agent:
    """High-level agent interface for easy agent creation.

    This is the recommended API for most use cases. It provides a simple,
    ergonomic interface while still allowing access to lower-level primitives
    when needed.

    Args:
        model: Either a model string (e.g., "gpt-4") or a provider instance
               (e.g., OpenAI(model="gpt-4"), VLLM(...), Anthropic(...))
        tools: Optional list of tool functions. Tools can be plain async functions
               or decorated with @tool for advanced features.
        max_iterations: Maximum number of agent loop iterations (default: 10)
        system_prompt: Optional system prompt to prepend to conversations
        temperature: LLM temperature parameter (default: 0.7)
        max_tokens: Maximum tokens in LLM response (default: 2048)
        **llm_kwargs: Additional arguments passed to the LLM provider

    Example:
        >>> from thinkagain.agents import Agent
        >>>
        >>> async def get_weather(city: str) -> str:
        ...     return f"The weather in {city} is sunny, 72°F"
        >>>
        >>> agent = Agent(model="gpt-4", tools=[get_weather])
        >>> state = await agent.run("What's the weather in San Francisco?")
        >>> print(state.messages[-1].content)
    """

    def __init__(
        self,
        model: str | LLM,
        tools: list[Callable] | None = None,
        max_iterations: int = 10,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **llm_kwargs: Any,
    ):
        self.provider = OpenAI(model=model) if isinstance(model, str) else model
        self.tools = list(tools) if tools else None
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.llm_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **llm_kwargs,
        }

    def _init_state(self, message: str | None, state: AgentState | None) -> AgentState:
        """Initialize agent state from message or existing state."""
        if state is None:
            if message is None:
                raise ValueError("Either message or state must be provided")

            messages = []
            if self.system_prompt:
                messages.append(Message(role="system", content=self.system_prompt))
            messages.append(Message(role="user", content=message))
            state = AgentState(messages=messages, metadata={})
        elif message is not None:
            state.messages.append(Message(role="user", content=message))

        state.metadata["iterations"] = 0
        return state

    async def run(
        self,
        message: str | None = None,
        state: AgentState | None = None,
    ) -> AgentState:
        """Run the agent on a message and return the final state.

        Args:
            message: User message to process (required if state is None)
            state: Optional existing agent state to continue from

        Returns:
            Final agent state

        Example:
            >>> state = await agent.run("What is 15 * 234?")
            >>> print(state.messages[-1].content)
        """
        state = self._init_state(message, state)
        return await run_agent(
            state,
            provider=self.provider,
            tools=self.tools,
            max_iterations=self.max_iterations,
            llm_kwargs=self.llm_kwargs,
        )


async def run_agent(
    state: AgentState,
    provider: LLM,
    tools: list[Callable] | None,
    max_iterations: int,
    llm_kwargs: dict[str, Any],
) -> AgentState:
    """Execute the agent loop for a given state."""
    for _ in range(max_iterations):
        state.metadata["iterations"] = state.metadata.get("iterations", 0) + 1

        ctx = call_llm(Context(state), provider=provider, tools=tools, **llm_kwargs)
        state = await ctx

        has_tool_calls = bool(state.messages[-1].tool_calls)
        if not has_tool_calls:
            state.metadata["stop_reason"] = "complete"
            return state

        if tools:
            state = await execute_tools(state, tools=tools)

    state.metadata["stop_reason"] = "max_iterations"
    return state

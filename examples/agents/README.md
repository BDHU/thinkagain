# ThinkAgain Agents

Minimal agent helpers for ThinkAgain.

## Install

```bash
pip install openai  # for vLLM or OpenAI
pip install anthropic  # for Anthropic
```

## Quick Start

ThinkAgain provides two levels of abstraction - choose what fits your use case:

### Level 1: High-Level Agent API (Recommended)

The simplest way to build agents. Just 3 lines of code:

```python
from thinkagain.agents import Agent, tool, VLLM

@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

# Create and run agent in 3 lines
agent = Agent(model=VLLM(model="llama-3.1-8b"), tools=[calculate])
state = await agent.run("What is 15 * 234?")
print(state.messages[-1].content)  # "3,510"
```

**Multi-turn conversations:**

```python
state = await agent.run("My name is Alice")
state = await agent.run("What's my name?", state=state)  # "Your name is Alice"
```

### Level 2: Low-Level Node API (Custom Control Flow)

For custom workflows and fine-grained control:

```python
from thinkagain import Context
from thinkagain.agents import AgentState, Message, call_llm, execute_tools, VLLM

@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

# Manual loop for maximum control
state = AgentState(messages=[Message(role="user", content="What is 15 * 234?")])
provider = VLLM(model="llama-3.1-8b")
tools = [calculate]

for _ in range(10):
    # call_llm is a @node that updates the state
    ctx = call_llm(Context(state), provider=provider, tools=tools)
    state = await ctx

    if not state.messages[-1].tool_calls:
        break

    # execute_tools runs tool calls and returns the updated state
    state = await execute_tools(state, tools=tools)
```

## Run Examples

Start a vLLM server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
```

Run the examples:

```bash
# High-level Agent API (recommended for beginners)
python examples/agents/simple_agent.py
```

## Core Components

### High-Level API
- **Agent**: Main agent class with async `.run()` returning the final state

### Low-Level Node API
- **State**: `AgentState`, `Message`, `ToolCall`
- **Nodes**: `call_llm`, `execute_tool_call` (both are @node decorated)
- **Decorators**: `@tool` (marks plain async functions as tools)
- **Providers**: `VLLM`, `OpenAI`, `Anthropic`

### Architecture
- **Tools** (`@tool`): Plain async functions, NOT `@node` decorated
- **Core nodes**: `call_llm` and `execute_tool_call` are @node units
- **Agent.run()**: Returns the final agent state

## Choose Your Level

- **Beginners / Quick tasks**: Use `Agent` class
- **Custom control flow**: Use manual `call_llm` + `execute_tools` nodes
- **Complex workflows**: Build custom graphs with @node primitives

Both levels work together - start simple and drop down to lower levels only when you need more control.

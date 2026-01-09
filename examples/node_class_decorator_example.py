"""Example demonstrating @node decorator on classes with async __call__.

This example shows how to use the @node decorator on classes, similar to JAX,
allowing for stateful computation nodes within @jit pipelines.
"""

import asyncio
from dataclasses import dataclass

import thinkagain


@dataclass
class State:
    """Simple state for demonstration."""

    value: int
    label: str = ""


# ============================================================================
# Example 1: Basic @node decorated class
# ============================================================================


@thinkagain.node
class Multiplier:
    """A stateful node that multiplies by a configured factor."""

    def __init__(self, factor: int):
        self.factor = factor

    async def __call__(self, state: State) -> State:
        """Multiply state value by the configured factor."""
        return State(
            value=state.value * self.factor, label=f"{state.label}_x{self.factor}"
        )


@thinkagain.jit
async def pipeline_with_multiplier(state: State) -> State:
    """Pipeline using a @node decorated class."""
    mult = Multiplier(3)
    return await mult(state)


# ============================================================================
# Example 2: @node class in control flow
# ============================================================================


@thinkagain.node
class Adder:
    """A stateful node that adds a configured amount."""

    def __init__(self, amount: int):
        self.amount = amount

    async def __call__(self, state: State) -> State:
        """Add the configured amount to state value."""
        return State(
            value=state.value + self.amount, label=f"{state.label}_+{self.amount}"
        )


@thinkagain.jit
async def pipeline_with_conditional(state: State) -> State:
    """Pipeline using @node classes in conditional branches."""
    add_small = Adder(5)
    add_large = Adder(20)

    # Use different adders based on state value
    return await thinkagain.cond(
        lambda s: s.value < 10,
        add_large,  # Add more to small values
        add_small,  # Add less to large values
        state,
    )


# ============================================================================
# Example 3: @node class in loops
# ============================================================================


@thinkagain.node
class Incrementer:
    """A stateful node that increments by a configured step."""

    def __init__(self, step: int = 1):
        self.step = step

    async def __call__(self, state: State) -> State:
        """Increment state value by step."""
        return State(value=state.value + self.step, label=state.label)


@thinkagain.jit
async def pipeline_with_while_loop(state: State) -> State:
    """Pipeline using @node class in while loop."""
    inc = Incrementer(2)
    return await thinkagain.while_loop(
        lambda s: s.value < 10,
        inc,
        state,
    )


# ============================================================================
# Example 4: @node class in scan
# ============================================================================


@thinkagain.node
class ListProcessor:
    """A stateful node for processing list items with scan."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    async def __call__(self, carry: int, item: str) -> tuple[int, str]:
        """Process each item, returning new carry and output."""
        new_carry = carry + 1
        output = f"{self.prefix}[{carry}]: {item.upper()}"
        return new_carry, output


@thinkagain.jit
async def pipeline_with_scan(items: list[str]) -> list[str]:
    """Pipeline using @node class in scan."""
    processor = ListProcessor("Item")
    _, results = await thinkagain.scan(processor, 0, items)
    return results


# ============================================================================
# Example 5: Multiple instances with different configurations
# ============================================================================


@thinkagain.node
class ConfigurableMultiplier:
    """Configurable node that can be instantiated with different settings."""

    def __init__(self, multiplier: int, label: str):
        self.multiplier = multiplier
        self.label = label

    async def __call__(self, state: State) -> State:
        return State(
            value=state.value * self.multiplier,
            label=f"{state.label}_{self.label}",
        )


@thinkagain.jit
async def pipeline_with_multiple_configs(state: State) -> State:
    """Pipeline using multiple instances of the same @node class."""
    double = ConfigurableMultiplier(2, "x2")
    triple = ConfigurableMultiplier(3, "x3")

    # Apply transformations sequentially
    state = await double(state)
    state = await triple(state)

    return state


# ============================================================================
# Demo
# ============================================================================


async def main():
    """Run all examples."""
    print("=" * 70)
    print("@node Decorator on Classes - Examples")
    print("=" * 70)

    # Example 1: Basic usage
    print("\n1. Basic @node decorated class:")
    result = await pipeline_with_multiplier(State(value=7))
    print("   Input: State(value=7)")
    print(f"   Output: {result}")

    # Example 2: Control flow
    print("\n2. @node class in conditional:")
    result_small = await pipeline_with_conditional(State(value=3))
    result_large = await pipeline_with_conditional(State(value=15))
    print(f"   Small value (3): {result_small}")
    print(f"   Large value (15): {result_large}")

    # Example 3: While loop
    print("\n3. @node class in while loop:")
    result = await pipeline_with_while_loop(State(value=0))
    print("   Input: State(value=0), target=10")
    print(f"   Output: {result}")

    # Example 4: Scan
    print("\n4. @node class in scan:")
    results = await pipeline_with_scan(["apple", "banana", "cherry"])
    print("   Input: ['apple', 'banana', 'cherry']")
    print("   Output:")
    for r in results:
        print(f"     - {r}")

    # Example 5: Multiple configurations
    print("\n5. Multiple instances with different configurations:")
    result = await pipeline_with_multiple_configs(State(value=5))
    print("   Input: State(value=5)")
    print(f"   Output: {result}")
    print("   Value: 5 × 2 × 3 = 30")

    print("\n" + "=" * 70)
    print("Summary:")
    print("  - @node decorator works on both functions and classes")
    print("  - Classes must implement async __call__ method")
    print("  - Supports all control flow operators (cond, while, scan, switch)")
    print("  - Allows stateful nodes with clean, JAX-like syntax")
    print("  - No need for base class inheritance!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

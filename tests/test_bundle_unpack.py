"""Test script for bundle.unpack() method."""

import asyncio
from dataclasses import dataclass

import pytest

import thinkagain as ta


@ta.trace
@dataclass
class SampleInputs:
    query: str
    db: str
    limit: int


@pytest.mark.asyncio
async def test_unpack_basic():
    """Test basic unpacking of Bundle."""
    print("Test 1: Basic unpacking from Bundle")
    inputs = ta.Bundle(query="search", db="postgres", limit=10)

    # Test unpacking
    query, db = await ta.bundle.unpack(inputs, "query", "db")

    assert query == "search", f"Expected 'search', got {query}"
    assert db == "postgres", f"Expected 'postgres', got {db}"
    print(f"  ✓ Unpacked: query={query}, db={db}")


@pytest.mark.asyncio
async def test_unpack_order():
    """Test that unpacking preserves order."""
    print("\nTest 2: Order preservation")
    inputs = ta.Bundle(a=1, b=2, c=3, d=4)

    # Unpack in different order
    d, b, a = await ta.bundle.unpack(inputs, "d", "b", "a")

    assert d == 4 and b == 2 and a == 1, f"Order not preserved: {d}, {b}, {a}"
    print(f"  ✓ Order preserved: d={d}, b={b}, a={a}")


@pytest.mark.asyncio
async def test_unpack_single():
    """Test unpacking single value (should still return tuple)."""
    print("\nTest 3: Single value unpacking")
    inputs = ta.Bundle(value=42)

    (value,) = await ta.bundle.unpack(inputs, "value")

    assert value == 42, f"Expected 42, got {value}"
    print(f"  ✓ Single value: {value}")


@pytest.mark.asyncio
async def test_unpack_all():
    """Test unpacking all values."""
    print("\nTest 4: Unpack all values")
    inputs = ta.Bundle(x=10, y=20, z=30)

    x, y, z = await ta.bundle.unpack(inputs, "x", "y", "z")

    assert x == 10 and y == 20 and z == 30, f"Values: {x}, {y}, {z}"
    print(f"  ✓ All values: x={x}, y={y}, z={z}")


@pytest.mark.asyncio
async def test_unpack_dataclass():
    """Test unpacking from traced dataclass."""
    print("\nTest 5: Unpacking from dataclass")
    inputs = SampleInputs(query="test", db="mysql", limit=5)

    query, limit = await ta.bundle.unpack(inputs, "query", "limit")

    assert query == "test", f"Expected 'test', got {query}"
    assert limit == 5, f"Expected 5, got {limit}"
    print(f"  ✓ Dataclass unpack: query={query}, limit={limit}")


@pytest.mark.asyncio
async def test_unpack_missing_key():
    """Test error handling for missing keys."""
    print("\nTest 6: Missing key error handling")
    inputs = ta.Bundle(a=1, b=2)

    try:
        await ta.bundle.unpack(inputs, "a", "missing", "b")
        print("  ✗ Should have raised KeyError")
        return False
    except KeyError as e:
        error_msg = str(e)
        assert "missing" in error_msg, f"Error should mention 'missing': {error_msg}"
        assert "Available" in error_msg, (
            f"Error should show available keys: {error_msg}"
        )
        print(f"  ✓ Correct error: {error_msg}")
        return True


@pytest.mark.asyncio
async def test_unpack_creates_graph_node():
    """Test that unpack creates a graph node."""
    print("\nTest 7: Graph node creation")

    @ta.node
    async def use_unpack(inputs):
        query, db = await ta.bundle.unpack(inputs, "query", "db")
        return f"{query}:{db}"

    inputs = ta.Bundle(query="select", db="redis")
    result = await use_unpack(inputs)

    assert result == "select:redis", f"Expected 'select:redis', got {result}"
    print(f"  ✓ Works in graph context: {result}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing ta.bundle.unpack()")
    print("=" * 60)

    try:
        await test_unpack_basic()
        await test_unpack_order()
        await test_unpack_single()
        await test_unpack_all()
        await test_unpack_dataclass()
        await test_unpack_missing_key()
        await test_unpack_creates_graph_node()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())

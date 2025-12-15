"""Tests for declarative graph construction API."""

import asyncio

import pytest

from thinkagain import Node, run, Context, Executable, LazyContext, NeedsMaterializationError


# -----------------------------------------------------------------------------
# Test Executables
# -----------------------------------------------------------------------------


class AddOne(Executable):
    async def arun(self, ctx: Context) -> Context:
        ctx.value = ctx.get("value", 0) + 1
        return ctx


class Double(Executable):
    async def arun(self, ctx: Context) -> Context:
        ctx.value = ctx.value * 2
        return ctx


class SetScore(Executable):
    def __init__(self, score: float):
        super().__init__()
        self.score = score

    async def arun(self, ctx: Context) -> Context:
        ctx.score = self.score
        return ctx


class AppendLog(Executable):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    async def arun(self, ctx: Context) -> Context:
        logs = ctx.get("logs", [])
        ctx.logs = logs + [self.message]
        return ctx


# -----------------------------------------------------------------------------
# Test: Basic Pipeline
# -----------------------------------------------------------------------------


def test_linear_pipeline():
    """Test simple linear pipeline with lazy execution."""
    add_one = Node(AddOne())
    double = Node(Double())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        ctx = await double(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {"value": 5}))

    assert result.value == 12  # (5 + 1) * 2


def test_empty_pipeline():
    """Test pipeline that does nothing."""

    async def pipeline(ctx):
        return ctx

    result = asyncio.run(run(pipeline, {"value": 42}))

    assert result.value == 42


def test_dict_input():
    """Test passing dict as initial context."""
    add_one = Node(AddOne())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {"value": 10}))

    assert result.value == 11


def test_context_input():
    """Test passing Context as initial context."""
    add_one = Node(AddOne())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        return ctx

    result = asyncio.run(run(pipeline, Context(value=10)))

    assert result.value == 11


def test_none_input():
    """Test passing None as initial context."""
    add_one = Node(AddOne())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        return ctx

    result = asyncio.run(run(pipeline, None))

    assert result.value == 1  # 0 + 1


# -----------------------------------------------------------------------------
# Test: Materialization
# -----------------------------------------------------------------------------


def test_explicit_materialization():
    """Test explicit materialization with await ctx."""
    add_one = Node(AddOne())
    double = Node(Double())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        ctx = await ctx  # materialize
        assert ctx.value == 6  # 5 + 1
        ctx = await double(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {"value": 5}))

    assert result.value == 12


def test_access_before_materialization_raises():
    """Test that accessing values before materialization raises error."""
    add_one = Node(AddOne())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        _ = ctx.value  # should raise
        return ctx

    with pytest.raises(NeedsMaterializationError):
        asyncio.run(run(pipeline, {"value": 5}))


def test_set_before_materialization_raises():
    """Test that setting values before materialization raises error."""
    add_one = Node(AddOne())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        ctx.foo = "bar"  # should raise
        return ctx

    with pytest.raises(NeedsMaterializationError):
        asyncio.run(run(pipeline, {"value": 5}))


def test_get_before_materialization_raises():
    """Test that get() before materialization raises error."""
    add_one = Node(AddOne())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        _ = ctx.get("value")  # should raise
        return ctx

    with pytest.raises(NeedsMaterializationError):
        asyncio.run(run(pipeline, {"value": 5}))


# -----------------------------------------------------------------------------
# Test: Conditional Branching
# -----------------------------------------------------------------------------


def test_if_else_branching():
    """Test conditional branching with if/else."""
    set_high = Node(SetScore(0.9))
    set_low = Node(SetScore(0.3))

    async def pipeline(ctx):
        if ctx.use_high:
            ctx = await set_high(ctx)
        else:
            ctx = await set_low(ctx)
        return ctx

    # Test high branch
    result = asyncio.run(run(pipeline, {"use_high": True}))
    assert result.score == 0.9

    # Test low branch
    result = asyncio.run(run(pipeline, {"use_high": False}))
    assert result.score == 0.3


def test_conditional_after_materialization():
    """Test conditional that depends on computed value."""
    add_one = Node(AddOne())
    log_high = Node(AppendLog("high"))
    log_low = Node(AppendLog("low"))

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        ctx = await ctx  # materialize to access value

        if ctx.value > 5:
            ctx = await log_high(ctx)
        else:
            ctx = await log_low(ctx)

        return ctx

    # value = 10 + 1 = 11 > 5
    result = asyncio.run(run(pipeline, {"value": 10}))
    assert result.logs == ["high"]

    # value = 2 + 1 = 3 < 5
    result = asyncio.run(run(pipeline, {"value": 2}))
    assert result.logs == ["low"]


# -----------------------------------------------------------------------------
# Test: Loops
# -----------------------------------------------------------------------------


def test_while_loop():
    """Test while loop with materialization each iteration."""
    add_one = Node(AddOne())

    async def pipeline(ctx):
        ctx = await ctx  # materialize initial

        while ctx.value < 5:
            ctx = await add_one(ctx)
            ctx = await ctx  # materialize for next check

        return ctx

    result = asyncio.run(run(pipeline, {"value": 0}))

    assert result.value == 5


def test_for_loop():
    """Test for loop with multiple node calls."""
    log = Node(AppendLog("x"))

    async def pipeline(ctx):
        for _ in range(3):
            ctx = await log(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {}))

    assert result.logs == ["x", "x", "x"]


# -----------------------------------------------------------------------------
# Test: Node Naming
# -----------------------------------------------------------------------------


def test_node_name_from_executable():
    """Test node name inference from Executable."""
    node = Node(AddOne())
    assert node.name == "addone"


def test_node_name_explicit():
    """Test explicit node name."""
    node = Node(AddOne(), name="my_adder")
    assert node.name == "my_adder"


def test_node_repr():
    """Test Node repr."""
    node = Node(AddOne(), name="adder")
    assert repr(node) == "Node('adder')"


# -----------------------------------------------------------------------------
# Test: LazyContext
# -----------------------------------------------------------------------------


def test_lazy_context_repr_pending():
    """Test LazyContext repr with pending nodes."""
    node = Node(AddOne(), name="add")
    ctx = LazyContext({"value": 1}, [node])
    assert "pending" in repr(ctx)
    assert "add" in repr(ctx)


def test_lazy_context_repr_materialized():
    """Test LazyContext repr when materialized."""
    ctx = LazyContext({"value": 1}, [])
    assert "value" in repr(ctx)


def test_lazy_context_is_pending():
    """Test is_pending property."""
    node = Node(AddOne())

    ctx = LazyContext({}, [])
    assert not ctx.is_pending

    ctx = LazyContext({}, [node])
    assert ctx.is_pending


# -----------------------------------------------------------------------------
# Test: Complex Pipelines
# -----------------------------------------------------------------------------


def test_nested_function_calls():
    """Test calling helper functions within pipeline."""
    add_one = Node(AddOne())
    double = Node(Double())

    async def add_twice(ctx):
        ctx = await add_one(ctx)
        ctx = await add_one(ctx)
        return ctx

    async def pipeline(ctx):
        ctx = await add_twice(ctx)
        ctx = await double(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {"value": 5}))

    assert result.value == 14  # (5 + 1 + 1) * 2


def test_early_return():
    """Test early return from pipeline."""
    add_one = Node(AddOne())
    double = Node(Double())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        ctx = await ctx

        if ctx.value > 10:
            return ctx  # early return

        ctx = await double(ctx)
        return ctx

    # value = 5 + 1 = 6, not > 10, so double
    result = asyncio.run(run(pipeline, {"value": 5}))
    assert result.value == 12

    # value = 15 + 1 = 16 > 10, early return
    result = asyncio.run(run(pipeline, {"value": 15}))
    assert result.value == 16


def test_multiple_materializations():
    """Test multiple materialization points."""
    add_one = Node(AddOne())

    async def pipeline(ctx):
        ctx = await add_one(ctx)
        ctx = await ctx
        assert ctx.value == 1

        ctx = await add_one(ctx)
        ctx = await ctx
        assert ctx.value == 2

        ctx = await add_one(ctx)
        ctx = await ctx
        assert ctx.value == 3

        return ctx

    result = asyncio.run(run(pipeline, {"value": 0}))
    assert result.value == 3

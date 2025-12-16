"""Tests for declarative API."""

import pytest
from thinkagain import Node, run, Context, node


# Test functions
@node
async def add_one(ctx):
    ctx.value = ctx.get("value", 0) + 1
    return ctx


@node
async def double(ctx):
    ctx.value = ctx.value * 2
    return ctx


def test_linear_pipeline():
    def pipeline(ctx):
        ctx = add_one(ctx)
        ctx = double(ctx)
        return ctx

    result = run(pipeline, {"value": 5})
    assert result.value == 12  # (5+1)*2


def test_empty_pipeline():
    def pipeline(ctx):
        return ctx

    result = run(pipeline, {"value": 42})
    assert result.value == 42


def test_dict_input():
    def pipeline(ctx):
        ctx = add_one(ctx)
        return ctx

    result = run(pipeline, {"value": 10})
    assert result.value == 11


def test_none_input():
    def pipeline(ctx):
        ctx = add_one(ctx)
        return ctx

    result = run(pipeline)
    assert result.value == 1


def test_auto_materialization():
    def pipeline(ctx):
        ctx = add_one(ctx)
        # Accessing .value auto-materializes
        assert ctx.value == 6
        ctx = double(ctx)
        return ctx

    result = run(pipeline, {"value": 5})
    assert result.value == 12


def test_if_else_branching():
    @node
    async def high(ctx):
        ctx.score = 0.9
        return ctx

    @node
    async def low(ctx):
        ctx.score = 0.3
        return ctx

    def pipeline(ctx):
        if ctx.use_high:  # auto-materializes to check
            ctx = high(ctx)
        else:
            ctx = low(ctx)
        return ctx

    assert run(pipeline, {"use_high": True}).score == 0.9
    assert run(pipeline, {"use_high": False}).score == 0.3


def test_conditional_after_node():
    @node
    async def log_high(ctx):
        ctx.logs = ["high"]
        return ctx

    @node
    async def log_low(ctx):
        ctx.logs = ["low"]
        return ctx

    def pipeline(ctx):
        ctx = add_one(ctx)
        # Accessing ctx.value auto-materializes
        if ctx.value > 5:
            ctx = log_high(ctx)
        else:
            ctx = log_low(ctx)
        return ctx

    assert run(pipeline, {"value": 10}).logs == ["high"]
    assert run(pipeline, {"value": 2}).logs == ["low"]


def test_while_loop():
    def pipeline(ctx):
        while ctx.value < 5:  # auto-materializes each iteration
            ctx = add_one(ctx)
        return ctx

    result = run(pipeline, {"value": 0})
    assert result.value == 5


def test_for_loop():
    @node
    async def append_x(ctx):
        ctx.logs = ctx.get("logs", []) + ["x"]
        return ctx

    def pipeline(ctx):
        for _ in range(3):
            ctx = append_x(ctx)
        return ctx

    result = run(pipeline, {})
    assert result.logs == ["x", "x", "x"]


def test_node_name():
    assert add_one.name == "add_one"


def test_node_explicit_name():
    @node
    async def foo(ctx):
        return ctx

    n = Node(foo.fn, name="custom")
    assert n.name == "custom"


def test_node_decorator_with_name():
    @node(name="custom_name")
    async def foo(ctx):
        return ctx

    assert isinstance(foo, Node)
    assert foo.name == "custom_name"


def test_context_is_pending():
    ctx = Context({})
    assert not ctx.is_pending
    ctx = add_one(ctx)  # chaining a node creates pending state
    assert ctx.is_pending


def test_nested_functions():
    def add_twice(ctx):
        ctx = add_one(ctx)
        ctx = add_one(ctx)
        return ctx

    def pipeline(ctx):
        ctx = add_twice(ctx)
        ctx = double(ctx)
        return ctx

    result = run(pipeline, {"value": 5})
    assert result.value == 14  # (5+1+1)*2


def test_early_return():
    def pipeline(ctx):
        ctx = add_one(ctx)
        if ctx.value > 10:  # auto-materializes
            return ctx
        ctx = double(ctx)
        return ctx

    assert run(pipeline, {"value": 5}).value == 12
    assert run(pipeline, {"value": 15}).value == 16


def test_direct_node_call():
    # Can call nodes directly without run()
    ctx = add_one({"value": 5})
    ctx = double(ctx)
    assert ctx.value == 12

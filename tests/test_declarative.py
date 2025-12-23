"""Tests for declarative API."""

from dataclasses import asdict, is_dataclass

import pytest

from thinkagain import Context, FunctionNode, Node, NodeDataclassError, run, node
from conftest import add_one, double


# =============================================================================
# Basic pipeline tests
# =============================================================================


@pytest.mark.parametrize(
    ("initial", "pipeline_fn", "expected"),
    [
        ({"value": 5}, lambda ctx, n: n["double"](n["add_one"](ctx)), 12),
        ({"value": 42}, lambda ctx, n: ctx, 42),  # empty pipeline
        ({"value": 10}, lambda ctx, n: n["add_one"](ctx), 11),
        ({}, lambda ctx, n: n["add_one"](ctx), 1),  # default value
    ],
)
def test_basic_pipelines(initial, pipeline_fn, expected, nodes):
    def pipeline(ctx):
        return pipeline_fn(ctx, nodes)

    result = run(pipeline, initial)
    assert result.get("value") == expected


def test_auto_materialization():
    def pipeline(ctx):
        ctx = add_one(ctx)
        assert ctx.get("value") == 6  # materializes
        ctx = double(ctx)
        return ctx

    result = run(pipeline, {"value": 5})
    assert result.get("value") == 12


# =============================================================================
# Branching tests
# =============================================================================


@pytest.mark.parametrize(
    ("use_high", "expected"),
    [(True, 0.9), (False, 0.3)],
)
def test_if_else_branching(use_high, expected):
    @node
    async def high(ctx):
        ctx.set("score", 0.9)
        return ctx

    @node
    async def low(ctx):
        ctx.set("score", 0.3)
        return ctx

    def pipeline(ctx):
        return high(ctx) if ctx.get("use_high") else low(ctx)

    assert run(pipeline, {"use_high": use_high}).get("score") == expected


@pytest.mark.parametrize(
    ("value", "expected_log"),
    [(10, ["high"]), (2, ["low"])],
)
def test_conditional_after_node(value, expected_log):
    @node
    async def log_high(ctx):
        ctx.set("logs", ["high"])
        return ctx

    @node
    async def log_low(ctx):
        ctx.set("logs", ["low"])
        return ctx

    def pipeline(ctx):
        ctx = add_one(ctx)
        return log_high(ctx) if ctx.get("value") > 5 else log_low(ctx)

    assert run(pipeline, {"value": value}).get("logs") == expected_log


# =============================================================================
# Loop tests
# =============================================================================


def test_while_loop():
    def pipeline(ctx):
        while ctx.get("value") < 5:
            ctx = add_one(ctx)
        return ctx

    assert run(pipeline, {"value": 0}).get("value") == 5


def test_for_loop():
    @node
    async def append_x(ctx):
        ctx.set("logs", ctx.get("logs", []) + ["x"])
        return ctx

    def pipeline(ctx):
        for _ in range(3):
            ctx = append_x(ctx)
        return ctx

    assert run(pipeline, {}).get("logs") == ["x", "x", "x"]


# =============================================================================
# Node naming tests
# =============================================================================


def test_node_names():
    assert add_one.name == "add_one"

    @node
    async def foo(ctx):
        return ctx

    assert FunctionNode(foo.fn, name="custom").name == "custom"

    @node(name="custom_name")
    async def bar(ctx):
        return ctx

    assert bar.name == "custom_name"


# =============================================================================
# Context tests
# =============================================================================


def test_context_is_pending():
    ctx = Context({})
    assert not ctx.is_pending
    ctx = add_one(ctx)
    assert ctx.is_pending


def test_nested_functions():
    def add_twice(ctx):
        return add_one(add_one(ctx))

    def pipeline(ctx):
        return double(add_twice(ctx))

    assert run(pipeline, {"value": 5}).get("value") == 14


def test_early_return():
    def pipeline(ctx):
        ctx = add_one(ctx)
        if ctx.get("value") > 10:
            return ctx
        return double(ctx)

    assert run(pipeline, {"value": 5}).get("value") == 12
    assert run(pipeline, {"value": 15}).get("value") == 16


def test_direct_node_call():
    ctx = double(add_one({"value": 5}))
    assert ctx.get("value") == 12


# =============================================================================
# Fanout and DAG tests
# =============================================================================


def test_fanout_deduplication():
    """Shared ancestor nodes execute only once."""
    call_count = {"n": 0}

    @node
    async def counting_node(ctx):
        call_count["n"] += 1
        ctx.set("value", ctx.get("value", 0) + 1)
        return ctx

    ctx = counting_node(Context({"value": 0}))
    ctx2, ctx3 = add_one(ctx), double(ctx)

    assert ctx2.get("value") == 2
    assert ctx3.get("value") == 2
    assert call_count["n"] == 1


def test_branch_data_isolation():
    """Branches get isolated copies of data."""
    ctx1 = add_one(Context({"value": 10}))
    ctx2, ctx3 = double(ctx1), add_one(ctx1)

    assert ctx2.get("value") == 22
    assert ctx3.get("value") == 12


def test_deep_fanout():
    """Fanout works with deeper graphs."""
    ctx = Context({"value": 1})
    ctx_b = double(add_one(ctx))
    ctx_c, ctx_d = add_one(ctx_b), double(ctx_b)

    ctx_c.materialize()
    ctx_d.materialize()

    assert ctx_c.get("value") == 5
    assert ctx_d.get("value") == 8


def test_pending_names_with_fanout():
    ctx = double(add_one(Context({"value": 0})))
    assert ctx.pending_names == ["add_one", "double"]


# =============================================================================
# Class-based Node tests
# =============================================================================


class AddValue(Node):
    delta: int = 1

    async def run(self, ctx):
        ctx.set("value", ctx.get("value", 0) + self.delta)
        return ctx


class Multiply(Node):
    factor: int = 2

    async def run(self, ctx):
        ctx.set("value", ctx.get("value", 0) * self.factor)
        return ctx


@pytest.mark.parametrize(
    ("delta", "initial", "expected"),
    [(5, 10, 15), (1, 10, 11)],  # explicit delta  # default delta
)
def test_class_node_basic(delta, initial, expected):
    adder = AddValue() if delta == 1 else AddValue(delta=delta)
    assert run(lambda ctx: adder(ctx), {"value": initial}).get("value") == expected


def test_class_node_chaining():
    adder, multiplier = AddValue(delta=3), Multiply(factor=2)

    def pipeline(ctx):
        return multiplier(adder(ctx))

    assert run(pipeline, {"value": 5}).get("value") == 16


def test_class_node_with_function_node():
    adder = AddValue(delta=10)

    def pipeline(ctx):
        return double(adder(add_one(ctx)))

    assert run(pipeline, {"value": 5}).get("value") == 32


def test_class_node_reusable():
    adder = AddValue(delta=1)

    def pipeline(ctx):
        return adder(adder(adder(ctx)))

    assert run(pipeline, {"value": 0}).get("value") == 3


def test_class_node_immutability():
    adder = AddValue(delta=5)
    with pytest.raises(AttributeError):
        adder.delta = 10  # type: ignore


def test_class_node_equality_and_hash():
    adder1, adder2, adder3 = AddValue(delta=5), AddValue(delta=5), AddValue(delta=10)

    assert adder1 == adder2
    assert adder1 != adder3
    assert len({adder1, adder2, adder3}) == 2


def test_class_node_serialization():
    adder = AddValue(delta=5)
    serialized = adder.serialize()
    assert serialized["type"] == "AddValue"
    assert serialized["config"] == {"delta": 5}
    assert asdict(adder) == {"delta": 5}


def test_class_node_auto_dataclass():
    class SimpleNode(Node):
        prefix: str = "hello"

        async def run(self, ctx):
            ctx.set("message", f"{self.prefix} world")
            return ctx

    assert is_dataclass(SimpleNode)
    assert SimpleNode.__dataclass_params__.frozen

    result = run(lambda ctx: SimpleNode(prefix="hi")(ctx), {})
    assert result.get("message") == "hi world"


def test_class_node_custom_init_rejected():
    with pytest.raises(NodeDataclassError):

        class BadNode(Node):
            def __init__(self, value: int) -> None:
                self.value = value

            async def run(self, ctx):
                return ctx


def test_class_node_explicit_dataclass_rejected():
    from dataclasses import dataclass

    with pytest.raises(TypeError):

        @dataclass(frozen=True)
        class SimpleNode(Node):
            prefix: str = "hello"

            async def run(self, ctx):
                return ctx


# =============================================================================
# Multi-input node tests
# =============================================================================


def test_multi_input_two_contexts():
    @node
    async def combine(ctx1, ctx2):
        return Context({"result": ctx1.get("a") + ctx2.get("b")})

    result = combine(Context({"a": 10}), Context({"b": 20}))
    assert result.get("result") == 30


def test_multi_input_with_value():
    @node
    async def scale(ctx, factor):
        return Context({"result": ctx.get("value") * factor})

    assert scale(Context({"value": 5}), 3).get("result") == 15


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.5, True), (0.9, False)],
)
def test_multi_input_with_kwarg(threshold, expected):
    @node
    async def process(ctx, threshold=0.5):
        return Context({"passed": ctx.get("value") > threshold})

    result = process(Context({"value": 0.7}), threshold=threshold)
    assert result.get("passed") is expected


def test_multi_input_shared_ancestor():
    call_count = {"n": 0}

    @node
    async def counting_node(ctx):
        call_count["n"] += 1
        ctx.set("value", ctx.get("value", 0) + 1)
        return ctx

    @node
    async def branch_a(ctx):
        ctx.set("a", ctx.get("value") * 2)
        return ctx

    @node
    async def branch_b(ctx):
        ctx.set("b", ctx.get("value") * 3)
        return ctx

    @node
    async def merge(ctx1, ctx2):
        return Context({"sum": ctx1.get("a") + ctx2.get("b")})

    ctx = counting_node(Context({"value": 0}))
    result = merge(branch_a(ctx), branch_b(ctx))

    assert result.get("sum") == 5
    assert call_count["n"] == 1


def test_multi_input_class_based_node():
    class Merger(Node):
        prefix: str = ""

        async def run(self, ctx1, ctx2):
            return Context(
                {"result": f"{self.prefix}{ctx1.get('a', '')}{ctx2.get('b', '')}"}
            )

    result = Merger(prefix="merged: ")(Context({"a": "hello"}), Context({"b": "world"}))
    assert result.get("result") == "merged: helloworld"


def test_multi_input_chained():
    @node
    async def add(ctx1, ctx2):
        return Context({"value": ctx1.get("value") + ctx2.get("value")})

    @node
    async def double_val(ctx):
        return Context({"value": ctx.get("value") * 2})

    result = double_val(add(Context({"value": 3}), Context({"value": 7})))
    assert result.get("value") == 20

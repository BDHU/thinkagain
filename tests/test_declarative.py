"""Tests for declarative API."""

from dataclasses import asdict, dataclass, is_dataclass
import pytest

from thinkagain import Context, FunctionNode, Node, NodeDataclassError, run, node


# Test functions
@node
async def add_one(ctx):
    ctx.set("value", ctx.get("value", 0) + 1)
    return ctx


@node
async def double(ctx):
    ctx.set("value", ctx.get("value") * 2)
    return ctx


def test_linear_pipeline():
    def pipeline(ctx):
        ctx = add_one(ctx)
        ctx = double(ctx)
        return ctx

    result = run(pipeline, {"value": 5})
    assert result.get("value") == 12  # (5+1)*2


def test_empty_pipeline():
    def pipeline(ctx):
        return ctx

    result = run(pipeline, {"value": 42})
    assert result.get("value") == 42


def test_dict_input():
    def pipeline(ctx):
        ctx = add_one(ctx)
        return ctx

    result = run(pipeline, {"value": 10})
    assert result.get("value") == 11


def test_none_input():
    def pipeline(ctx):
        ctx = add_one(ctx)
        return ctx

    result = run(pipeline)
    assert result.get("value") == 1


def test_auto_materialization():
    def pipeline(ctx):
        ctx = add_one(ctx)
        # Reading value materializes
        assert ctx.get("value") == 6
        ctx = double(ctx)
        return ctx

    result = run(pipeline, {"value": 5})
    assert result.get("value") == 12


def test_if_else_branching():
    @node
    async def high(ctx):
        ctx.set("score", 0.9)
        return ctx

    @node
    async def low(ctx):
        ctx.set("score", 0.3)
        return ctx

    def pipeline(ctx):
        if ctx.get("use_high"):  # materializes to check
            ctx = high(ctx)
        else:
            ctx = low(ctx)
        return ctx

    assert run(pipeline, {"use_high": True}).get("score") == 0.9
    assert run(pipeline, {"use_high": False}).get("score") == 0.3


def test_conditional_after_node():
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
        # Reading ctx value materializes
        if ctx.get("value") > 5:
            ctx = log_high(ctx)
        else:
            ctx = log_low(ctx)
        return ctx

    assert run(pipeline, {"value": 10}).get("logs") == ["high"]
    assert run(pipeline, {"value": 2}).get("logs") == ["low"]


def test_while_loop():
    def pipeline(ctx):
        while ctx.get("value") < 5:
            ctx = add_one(ctx)
        return ctx

    result = run(pipeline, {"value": 0})
    assert result.get("value") == 5


def test_for_loop():
    @node
    async def append_x(ctx):
        ctx.set("logs", ctx.get("logs", []) + ["x"])
        return ctx

    def pipeline(ctx):
        for _ in range(3):
            ctx = append_x(ctx)
        return ctx

    result = run(pipeline, {})
    assert result.get("logs") == ["x", "x", "x"]


def test_node_name():
    assert add_one.name == "add_one"


def test_node_explicit_name():
    @node
    async def foo(ctx):
        return ctx

    n = FunctionNode(foo.fn, name="custom")
    assert n.name == "custom"


def test_node_decorator_with_name():
    @node(name="custom_name")
    async def foo(ctx):
        return ctx

    assert isinstance(foo, FunctionNode)
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
    assert result.get("value") == 14  # (5+1+1)*2


def test_early_return():
    def pipeline(ctx):
        ctx = add_one(ctx)
        if ctx.get("value") > 10:
            return ctx
        ctx = double(ctx)
        return ctx

    assert run(pipeline, {"value": 5}).get("value") == 12
    assert run(pipeline, {"value": 15}).get("value") == 16


def test_direct_node_call():
    # Can call nodes directly without run()
    ctx = add_one({"value": 5})
    ctx = double(ctx)
    assert ctx.get("value") == 12


def test_context_metadata_records_timings():
    def pipeline(ctx):
        ctx = add_one(ctx)
        ctx = double(ctx)
        return ctx

    result = run(pipeline, {"value": 2})
    metadata = result.metadata
    assert metadata.total_duration >= 0
    assert [name for name, _ in metadata.node_latencies] == [
        "add_one",
        "double",
    ]
    assert metadata.per_node_totals["add_one"] >= 0
    assert metadata.per_node_totals["double"] >= 0
    assert metadata.finished_at is not None


def test_metadata_isolated_on_execution():
    """Metadata is shared during chaining but copied on execution."""
    ctx = run(lambda c: add_one(c), {"value": 1})
    finished_at = ctx.metadata.finished_at
    assert finished_at is not None
    original_count = ctx.metadata.node_execution_count

    # Chain a new node - metadata is shared until execution
    ctx2 = add_one(ctx)
    assert ctx2.is_pending  # not yet executed

    # After execution, new ctx has its own metadata with updated timings
    ctx2 = run(lambda c: c, ctx2)
    assert ctx2.metadata.node_execution_count == original_count + 1
    assert ctx2.metadata.finished_at is not None


# =============================================================================
# Class-based Node tests
# =============================================================================


class AddValue(Node):
    """A class-based node that adds a value."""

    delta: int = 1

    async def run(self, ctx):
        ctx.set("value", ctx.get("value", 0) + self.delta)
        return ctx


class Multiply(Node):
    """A class-based node that multiplies."""

    factor: int = 2

    async def run(self, ctx):
        ctx.set("value", ctx.get("value", 0) * self.factor)
        return ctx


def test_class_node_basic():
    """Class-based nodes work in pipelines."""
    adder = AddValue(delta=5)

    def pipeline(ctx):
        ctx = adder(ctx)
        return ctx

    result = run(pipeline, {"value": 10})
    assert result.get("value") == 15


def test_class_node_default_values():
    """Class-based nodes use default values."""
    adder = AddValue()  # delta defaults to 1

    result = run(lambda ctx: adder(ctx), {"value": 10})
    assert result.get("value") == 11


def test_class_node_chaining():
    """Multiple class-based nodes can be chained."""
    adder = AddValue(delta=3)
    multiplier = Multiply(factor=2)

    def pipeline(ctx):
        ctx = adder(ctx)
        ctx = multiplier(ctx)
        return ctx

    result = run(pipeline, {"value": 5})
    assert result.get("value") == 16  # (5+3)*2


def test_class_node_with_function_node():
    """Class-based nodes work with function-based nodes."""
    adder = AddValue(delta=10)

    def pipeline(ctx):
        ctx = add_one(ctx)  # function node
        ctx = adder(ctx)  # class node
        ctx = double(ctx)  # function node
        return ctx

    result = run(pipeline, {"value": 5})
    assert result.get("value") == 32  # ((5+1)+10)*2


def test_class_node_reusable():
    """Same node instance can be used multiple times."""
    adder = AddValue(delta=1)

    def pipeline(ctx):
        ctx = adder(ctx)
        ctx = adder(ctx)
        ctx = adder(ctx)
        return ctx

    result = run(pipeline, {"value": 0})
    assert result.get("value") == 3


def test_class_node_different_configs():
    """Different instances can have different configs."""
    small_add = AddValue(delta=1)
    big_add = AddValue(delta=100)

    def pipeline(ctx):
        ctx = small_add(ctx)
        ctx = big_add(ctx)
        return ctx

    result = run(pipeline, {"value": 0})
    assert result.get("value") == 101


def test_class_node_is_frozen():
    """Class-based nodes are immutable (frozen dataclass)."""
    adder = AddValue(delta=5)
    with pytest.raises(AttributeError):
        adder.delta = 10  # type: ignore


def test_class_node_name():
    """Class-based nodes use class name as node name."""
    adder = AddValue(delta=5)
    assert adder.name == "AddValue"


def test_class_node_equality():
    """Class-based nodes with same config are equal."""
    adder1 = AddValue(delta=5)
    adder2 = AddValue(delta=5)
    adder3 = AddValue(delta=10)

    assert adder1 == adder2
    assert adder1 != adder3


def test_class_node_hashable():
    """Class-based nodes are hashable (can be used in sets/dicts)."""
    adder1 = AddValue(delta=5)
    adder2 = AddValue(delta=5)
    adder3 = AddValue(delta=10)

    node_set = {adder1, adder2, adder3}
    assert len(node_set) == 2  # adder1 == adder2


def test_class_node_serialization():
    """Class-based nodes can be serialized."""
    adder = AddValue(delta=5)
    serialized = adder.serialize()

    assert serialized["type"] == "AddValue"
    assert serialized["config"] == {"delta": 5}


def test_class_node_asdict():
    """Class-based nodes support dataclass asdict."""
    adder = AddValue(delta=5)
    assert asdict(adder) == {"delta": 5}


def test_class_node_auto_dataclass():
    """Node subclasses are automatically frozen dataclasses."""

    class SimpleNode(Node):
        prefix: str = "hello"

        async def run(self, ctx):
            ctx.set("message", f"{self.prefix} world")
            return ctx

    assert is_dataclass(SimpleNode)
    assert SimpleNode.__dataclass_params__.frozen

    node_instance = SimpleNode(prefix="hi")
    result = run(lambda ctx: node_instance(ctx), {})
    assert result.get("message") == "hi world"


def test_class_node_custom_init_rejected():
    """Custom __init__ is rejected to avoid dataclass conflicts."""
    with pytest.raises(NodeDataclassError):

        class BadNode(Node):
            def __init__(self, value: int) -> None:
                self.value = value

            async def run(self, ctx):
                ctx.set("value", self.value)
                return ctx


def test_class_node_explicit_dataclass_is_rejected():
    """Explicit @dataclass on Node subclasses is not supported."""
    with pytest.raises(TypeError):

        @dataclass(frozen=True)
        class SimpleNode(Node):
            prefix: str = "hello"

            async def run(self, ctx):
                ctx.set("message", f"{self.prefix} world")
                return ctx


def test_class_node_metadata_recorded():
    """Class-based nodes are recorded in metadata."""
    adder = AddValue(delta=1)
    multiplier = Multiply(factor=2)

    def pipeline(ctx):
        ctx = adder(ctx)
        ctx = multiplier(ctx)
        return ctx

    result = run(pipeline, {"value": 5})
    metadata = result.metadata

    assert [name for name, _ in metadata.node_latencies] == ["AddValue", "Multiply"]


def test_class_node_direct_call():
    """Class-based nodes can be called directly without run()."""
    adder = AddValue(delta=5)
    multiplier = Multiply(factor=2)

    ctx = adder({"value": 10})
    ctx = multiplier(ctx)

    assert ctx.get("value") == 30  # (10+5)*2

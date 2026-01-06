"""Tests for declarative API."""

from dataclasses import asdict, dataclass, field, is_dataclass

import pytest

from thinkagain import (
    Context,
    FunctionNode,
    Node,
    NodeDataclassError,
    Sequential,
    run,
    node,
)
from conftest import add_one, double


# =============================================================================
# Basic pipeline tests
# =============================================================================


@pytest.mark.parametrize(
    ("initial", "pipeline_fn", "expected"),
    [
        (5, lambda ctx, n: n["double"](n["add_one"](ctx)), 12),
        (42, lambda ctx, n: ctx, 42),  # empty pipeline
        (10, lambda ctx, n: n["add_one"](ctx), 11),
        (0, lambda ctx, n: n["add_one"](ctx), 1),  # zero value
    ],
)
def test_basic_pipelines(initial, pipeline_fn, expected, nodes):
    def pipeline(ctx):
        return pipeline_fn(ctx, nodes)

    result = run(pipeline, initial)
    assert result.data == expected


def test_auto_materialization():
    def pipeline(ctx):
        ctx = add_one(ctx)
        assert ctx.data == 6  # materializes
        ctx = double(ctx)
        return ctx

    result = run(pipeline, 5)
    assert result.data == 12


# =============================================================================
# Branching tests
# =============================================================================


@pytest.mark.parametrize(
    ("use_high", "expected"),
    [(True, 0.9), (False, 0.3)],
)
def test_if_else_branching(use_high, expected):
    @node
    async def high(x) -> float:
        return 0.9

    @node
    async def low(x) -> float:
        return 0.3

    def pipeline(ctx):
        # ctx.data is a dict here with "use_high" key
        return high(ctx) if ctx.data["use_high"] else low(ctx)

    assert run(pipeline, {"use_high": use_high}).data == expected


@pytest.mark.parametrize(
    ("value", "expected_log"),
    [(10, ["high"]), (2, ["low"])],
)
def test_conditional_after_node(value, expected_log):
    @node
    async def log_high(x) -> list:
        return ["high"]

    @node
    async def log_low(x) -> list:
        return ["low"]

    def pipeline(ctx):
        ctx = add_one(ctx)
        return log_high(ctx) if ctx.data > 5 else log_low(ctx)

    assert run(pipeline, value).data == expected_log


# =============================================================================
# Loop tests
# =============================================================================


def test_while_loop():
    def pipeline(ctx):
        while ctx.data < 5:
            ctx = add_one(ctx)
        return ctx

    assert run(pipeline, 0).data == 5


def test_for_loop():
    @node
    async def append_x(logs: list) -> list:
        return logs + ["x"]

    def pipeline(ctx):
        for _ in range(3):
            ctx = append_x(ctx)
        return ctx

    assert run(pipeline, []).data == ["x", "x", "x"]


# =============================================================================
# Node naming tests
# =============================================================================


def test_node_names():
    assert add_one.name == "add_one"

    @node
    async def foo(x):
        return x

    assert FunctionNode(foo.fn, name="custom").name == "custom"

    @node(name="custom_name")
    async def bar(x):
        return x

    assert bar.name == "custom_name"


# =============================================================================
# Context tests
# =============================================================================


def test_context_is_pending():
    ctx = Context(5)
    assert not ctx.is_pending
    ctx = add_one(ctx)
    assert ctx.is_pending


def test_nested_functions():
    def add_twice(ctx):
        return add_one(add_one(ctx))

    def pipeline(ctx):
        return double(add_twice(ctx))

    assert run(pipeline, 5).data == 14


def test_early_return():
    def pipeline(ctx):
        ctx = add_one(ctx)
        if ctx.data > 10:
            return ctx
        return double(ctx)

    assert run(pipeline, 5).data == 12
    assert run(pipeline, 15).data == 16


def test_direct_node_call():
    ctx = double(add_one(Context(5)))
    assert ctx.data == 12


def test_node_requires_context_input():
    @node
    async def add(x: int) -> int:
        return x + 1

    with pytest.raises(TypeError):
        add(1)


# =============================================================================
# Fanout and DAG tests
# =============================================================================


def test_fanout_deduplication():
    """Shared ancestor nodes execute only once."""
    call_count = {"n": 0}

    @node
    async def counting_node(x: int) -> int:
        call_count["n"] += 1
        return x + 1

    ctx = counting_node(Context(0))
    ctx2, ctx3 = add_one(ctx), double(ctx)

    assert ctx2.data == 2
    assert ctx3.data == 2
    assert call_count["n"] == 1


def test_branch_data_isolation():
    """Branches get isolated copies of data."""
    ctx1 = add_one(Context(10))
    ctx2, ctx3 = double(ctx1), add_one(ctx1)

    assert ctx2.data == 22
    assert ctx3.data == 12


def test_deep_fanout():
    """Fanout works with deeper graphs."""
    ctx = Context(1)
    ctx_b = double(add_one(ctx))
    ctx_c, ctx_d = add_one(ctx_b), double(ctx_b)

    assert ctx_c.data == 5
    assert ctx_d.data == 8


def test_pending_names_with_fanout():
    ctx = double(add_one(Context(0)))
    assert ctx.pending_names == ["add_one", "double"]


# =============================================================================
# Class-based Node tests
# =============================================================================


class AddValue(Node):
    delta: int = 1

    async def run(self, x: int) -> int:
        return x + self.delta


class Multiply(Node):
    factor: int = 2

    async def run(self, x: int) -> int:
        return x * self.factor


@pytest.mark.parametrize(
    ("delta", "initial", "expected"),
    [(5, 10, 15), (1, 10, 11)],  # explicit delta  # default delta
)
def test_class_node_basic(delta, initial, expected):
    adder = AddValue() if delta == 1 else AddValue(delta=delta)
    assert run(lambda ctx: adder(ctx), initial).data == expected


def test_class_node_chaining():
    adder, multiplier = AddValue(delta=3), Multiply(factor=2)

    def pipeline(ctx):
        return multiplier(adder(ctx))

    assert run(pipeline, 5).data == 16


def test_class_node_with_function_node():
    adder = AddValue(delta=10)

    def pipeline(ctx):
        return double(adder(add_one(ctx)))

    assert run(pipeline, 5).data == 32


def test_class_node_reusable():
    adder = AddValue(delta=1)

    def pipeline(ctx):
        return adder(adder(adder(ctx)))

    assert run(pipeline, 0).data == 3


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

        async def run(self, x) -> str:
            return f"{self.prefix} world"

    assert is_dataclass(SimpleNode)
    assert SimpleNode.__dataclass_params__.frozen

    result = run(lambda ctx: SimpleNode(prefix="hi")(ctx), None)
    assert result.data == "hi world"


def test_class_node_custom_init_rejected():
    with pytest.raises(NodeDataclassError):

        class BadNode(Node):
            def __init__(self, value: int) -> None:
                self.value = value

            async def run(self, x):
                return x


def test_class_node_explicit_dataclass_rejected():
    from dataclasses import dataclass

    with pytest.raises(TypeError):

        @dataclass(frozen=True)
        class SimpleNode(Node):
            prefix: str = "hello"

            async def run(self, x):
                return x


# =============================================================================
# Multi-input node tests
# =============================================================================


def test_multi_input_two_contexts():
    @node
    async def combine(a: int, b: int) -> int:
        return a + b

    result = combine(Context(10), Context(20))
    assert result.data == 30


def test_multi_input_with_value():
    @node
    async def scale(value: int, factor: int) -> int:
        return value * factor

    assert scale(Context(5), 3).data == 15


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.5, True), (0.9, False)],
)
def test_multi_input_with_kwarg(threshold, expected):
    @node
    async def process(value: float, threshold: float = 0.5) -> bool:
        return value > threshold

    result = process(Context(0.7), threshold=threshold)
    assert result.data is expected


def test_multi_input_shared_ancestor():
    call_count = {"n": 0}

    @node
    async def counting_node(x: int) -> int:
        call_count["n"] += 1
        return x + 1

    @node
    async def branch_a(x: int) -> int:
        return x * 2

    @node
    async def branch_b(x: int) -> int:
        return x * 3

    @node
    async def merge(a: int, b: int) -> int:
        return a + b

    ctx = counting_node(Context(0))
    result = merge(branch_a(ctx), branch_b(ctx))

    assert result.data == 5
    assert call_count["n"] == 1


def test_multi_input_class_based_node():
    class Merger(Node):
        prefix: str = ""

        async def run(self, a: str, b: str) -> str:
            return f"{self.prefix}{a}{b}"

    result = Merger(prefix="merged: ")(Context("hello"), Context("world"))
    assert result.data == "merged: helloworld"


def test_multi_input_chained():
    @node
    async def add(a: int, b: int) -> int:
        return a + b

    @node
    async def double_val(x: int) -> int:
        return x * 2

    result = double_val(add(Context(3), Context(7)))
    assert result.data == 20


# =============================================================================
# Structured data tests (user manages structure)
# =============================================================================


def test_dict_as_data():
    """User can use dict as the data type."""

    @node
    async def add_field(data: dict) -> dict:
        return {**data, "added": True}

    ctx = Context({"existing": "value"})
    result = add_field(ctx)
    assert result.data == {"existing": "value", "added": True}


def test_dataclass_as_data():
    """User can use dataclass as the data type."""

    @dataclass
    class State:
        value: int
        processed: bool = False

    @node
    async def process_state(s: State) -> State:
        return State(value=s.value * 2, processed=True)

    ctx = Context(State(value=5))
    result = process_state(ctx)
    assert result.data == State(value=10, processed=True)


def test_pipeline_with_structured_data():
    """Full pipeline with structured data."""

    @dataclass
    class PipelineState:
        query: str
        docs: list = field(default_factory=list)
        answer: str = None

    @node
    async def retrieve(s: PipelineState) -> PipelineState:
        return PipelineState(query=s.query, docs=["doc1", "doc2"])

    @node
    async def generate(s: PipelineState) -> PipelineState:
        return PipelineState(query=s.query, docs=s.docs, answer=f"Answer for {s.query}")

    def pipeline(ctx):
        ctx = retrieve(ctx)
        ctx = generate(ctx)
        return ctx

    result = run(pipeline, PipelineState(query="What is X?"))
    assert result.data.query == "What is X?"
    assert result.data.docs == ["doc1", "doc2"]
    assert result.data.answer == "Answer for What is X?"


# =============================================================================
# Shared test state and nodes
# =============================================================================


@dataclass
class State:
    value: int
    name: str = "default"


@node
async def increment(s: State) -> State:
    return State(value=s.value + 1, name=s.name)


@node
async def double_state(s: State) -> State:
    return State(value=s.value * 2, name=s.name)


@node
async def square(s: State) -> State:
    return State(value=s.value**2)


# =============================================================================
# Context[T] generic typing tests
# =============================================================================


def test_context_generic_data_access():
    """Test that Context[T].data returns T with proper typing."""
    ctx: Context[State] = Context(State(value=42, name="test"))
    state = ctx.data
    assert isinstance(state, State)
    assert state.value == 42
    assert state.name == "test"


def test_context_generic_with_nodes():
    """Test Context[T] through node pipeline."""
    ctx: Context[State] = Context(State(value=5))
    ctx = increment(ctx)
    ctx = double_state(ctx)
    result: State = ctx.data
    assert result.value == 12  # (5 + 1) * 2


@pytest.mark.asyncio
async def test_context_async_data_access():
    """Test that async access also returns typed data."""
    ctx: Context[State] = increment(Context(State(value=5)))
    result: State = await ctx
    assert result.value == 6


def test_context_preserves_type_through_chain():
    """Test that type is preserved through multiple transformations."""

    @node
    async def transform1(s: State) -> State:
        return State(value=s.value + 1, name=s.name + "_1")

    @node
    async def transform2(s: State) -> State:
        return State(value=s.value * 2, name=s.name + "_2")

    @node
    async def transform3(s: State) -> State:
        return State(value=s.value - 3, name=s.name + "_3")

    ctx: Context[State] = Context(State(value=10, name="start"))
    ctx = transform1(ctx)
    ctx = transform2(ctx)
    ctx = transform3(ctx)

    result: State = ctx.data
    assert result.value == 19  # ((10 + 1) * 2) - 3
    assert result.name == "start_1_2_3"


# =============================================================================
# Sequential container tests
# =============================================================================


def test_sequential_basic():
    """Test basic Sequential pipeline."""
    pipeline = Sequential(add_one, double)
    ctx = pipeline(Context(5))
    assert ctx.data == 12  # (5 + 1) * 2


def test_sequential_with_dataclass():
    """Test Sequential with dataclass state."""
    pipeline = Sequential(increment, double_state, square)
    ctx = pipeline(Context(State(value=2)))
    assert ctx.data.value == 36  # ((2 + 1) * 2)^2 = 6^2


def test_sequential_composition():
    """Test composing and flattening Sequential pipelines."""
    stage1 = Sequential(increment, double_state)
    stage2 = Sequential(square)
    combined = Sequential(stage1, stage2)

    # Should flatten: increment, double_state, square
    assert len(combined) == 3
    assert combined.node_names == ["increment", "double_state", "square"]

    ctx = combined(Context(State(value=2)))
    assert ctx.data.value == 36  # ((2 + 1) * 2)^2


def test_sequential_iteration():
    """Test Sequential length, indexing, and iteration."""
    pipeline = Sequential(add_one, double, add_one)

    # Length
    assert len(pipeline) == 3

    # Indexing
    assert pipeline[0] == add_one
    assert pipeline[1] == double
    assert pipeline[-1] == add_one

    # Iteration
    steps = list(pipeline)
    assert steps == [add_one, double, add_one]

    # Node names
    assert pipeline.node_names == ["add_one", "double", "add_one"]


def test_sequential_repr():
    """Test Sequential string representation."""
    short = Sequential(add_one, double)
    assert repr(short) == "Sequential(add_one, double)"

    long = Sequential(add_one, double, add_one, add_one, double)
    assert "Sequential(add_one, ..., double) [5 steps]" == repr(long)


def test_sequential_reusability():
    """Test that Sequential pipelines are reusable."""
    pipeline = Sequential(add_one, double)

    ctx1 = pipeline(Context(5))
    ctx2 = pipeline(Context(10))

    assert ctx1.data == 12  # (5 + 1) * 2
    assert ctx2.data == 22  # (10 + 1) * 2


def test_sequential_in_function():
    """Test using Sequential in a function with conditional logic."""

    def custom_pipeline(ctx: Context[State]) -> Context[State]:
        ctx = Sequential(increment, double_state)(ctx)
        if ctx.data.value > 10:
            ctx = square(ctx)
        return ctx

    ctx = custom_pipeline(Context(State(value=5)))
    assert ctx.data.value == 144  # (5+1)*2 = 12, then 12^2 = 144

    ctx2 = custom_pipeline(Context(State(value=2)))
    assert ctx2.data.value == 6  # (2+1)*2 = 6, no square since <= 10

"""Slimmed down tests for distributed execution."""

import pytest

from thinkagain import Context, chain, node, replica, run
from thinkagain.distributed import get_default_manager, runtime
from thinkagain.distributed.profiling import (
    disable_profiling,
    enable_profiling,
    get_profiler,
    is_profiling_enabled,
    profiling_enabled,
)


def test_replica_registration_and_deploy():
    @replica(n=2)
    class Service:
        def __init__(self, value: int = 0):
            self.value = value

    spec = get_default_manager().get_spec("Service")
    assert spec.cls is Service.cls
    assert spec.n == 2

    Service.deploy(value=5)
    # Get instances via backend
    inst1 = Service.get()
    inst2 = Service.get()
    assert inst1.value == 5
    assert inst2.value == 5

    Service.shutdown()
    # After shutdown, get() auto-deploys with stored args
    assert Service.get().value == 5


def test_replica_round_robin():
    @replica(n=2)
    class Multiplier:
        def __init__(self, factor: int = 1):
            self.factor = factor
            self.call_count = 0

        def apply(self, x: int) -> int:
            self.call_count += 1
            return x * self.factor

    # Deploy two instances with different factors
    Multiplier.deploy(factor=2)

    # Round-robin should alternate between instances
    results = [Multiplier.get().apply(10) for _ in range(4)]
    assert results == [20, 20, 20, 20]  # Same factor, all 20


def test_replica_round_robin_distribution():
    """Test that round-robin actually distributes calls across different instances."""
    call_log = []

    @replica(n=2)
    class Logger:
        def __init__(self):
            pass

        def log(self):
            call_log.append(id(self))

    Logger.deploy()

    for _ in range(4):
        Logger.get().log()

    # Should see two distinct instances
    assert len(set(call_log)) == 2
    # Round-robin should alternate: [A, B, A, B]
    assert call_log[0] == call_log[2]
    assert call_log[1] == call_log[3]
    assert call_log[0] != call_log[1]


def test_pipeline_runs_with_replica():
    @replica
    class Adder:
        def __init__(self, delta: int = 1):
            self.delta = delta

        def apply(self, x: int) -> int:
            return x + self.delta

    @node
    async def increment(x: int) -> int:
        return x + 1

    @node
    async def apply_replica(x: int) -> int:
        return Adder.get().apply(x)

    pipeline = chain(increment, apply_replica)

    Adder.deploy(delta=5)
    result = run(pipeline, 3)
    assert result.data == 9

    Adder.shutdown()
    # After shutdown, get() lazily re-deploys with stored args (delta=5)
    assert Adder.get().apply(0) == 5


def test_runtime_context_manager():
    @replica(n=2)
    class Service:
        def ping(self) -> bool:
            return True

    @node
    async def call_service(_) -> bool:
        return Service.get().ping()

    pipeline = chain(call_service)

    with runtime():
        result = run(pipeline, None)
        assert result.data is True


def test_local_init_customizes_initialization():
    """Test that __local_init__ is called to customize instance creation."""
    init_calls = []

    @replica(n=2)
    class CustomInit:
        @classmethod
        def __local_init__(cls, base_value: int):
            # Simulate instance-specific customization
            init_calls.append(base_value)
            # Could inspect local resources here (memory, GPU, etc.)
            adjusted_value = base_value * 2
            return cls(adjusted_value)

        def __init__(self, value: int):
            self.value = value

        def get_value(self) -> int:
            return self.value

    CustomInit.deploy(base_value=10)

    # __local_init__ should have been called twice (n=2)
    assert len(init_calls) == 2
    assert init_calls == [10, 10]

    # Instances should have the adjusted value
    inst1 = CustomInit.get()
    inst2 = CustomInit.get()
    assert inst1.value == 20
    assert inst2.value == 20


def test_local_init_fallback_to_regular_init():
    """Test that classes without __local_init__ use regular __init__."""

    @replica(n=2)
    class RegularInit:
        def __init__(self, value: int):
            self.value = value

    RegularInit.deploy(value=5)

    inst1 = RegularInit.get()
    inst2 = RegularInit.get()
    assert inst1.value == 5
    assert inst2.value == 5


def test_profiling_toggle_and_context_manager():
    assert not is_profiling_enabled()
    assert get_profiler() is None

    profiler = enable_profiling()
    assert is_profiling_enabled()
    assert get_profiler() is profiler

    disable_profiling()
    assert not is_profiling_enabled()
    assert get_profiler() is None

    with profiling_enabled() as session:
        assert is_profiling_enabled()
        assert get_profiler() is session.profiler

    assert not is_profiling_enabled()


def test_profiling_dependency_fanout_counts_external_calls(shutdown_on_exit):
    @replica(n=2)
    class VectorDB:
        def search(self, query: str) -> list:
            return [f"result_{query}"]

    @replica(n=4)
    class LLMPool:
        def generate(self, context: str) -> str:
            return f"response_{context}"

    @replica(n=1)
    class Cache:
        def lookup(self, key: str) -> str | None:
            return None

    @replica
    class ExternalService:
        def ping(self) -> bool:
            return True

    @node
    async def retrieve(query: str) -> list:
        return VectorDB.get().search(query)

    @node
    async def generate(results: list) -> str:
        cached = Cache.get().lookup(str(results))
        if cached:
            return cached
        return LLMPool.get().generate(str(results))

    @node
    async def finalize(response: str) -> str:
        return LLMPool.get().generate(f"finalize_{response}")

    pipeline = chain(retrieve, generate, finalize)

    VectorDB.deploy()
    LLMPool.deploy()
    Cache.deploy()
    ExternalService.deploy()

    with shutdown_on_exit(VectorDB, LLMPool, Cache, ExternalService):
        with profiling_enabled() as session:
            result = run(
                pipeline,
                "test_query",
                context_factory=session.context_factory,
            )
            assert "finalize_response" in result.data

            ExternalService.get().ping()

            deps = session.profiler.get_dependency_graph()
            assert deps == {
                "retrieve": {"VectorDB"},
                "generate": {"Cache", "LLMPool"},
                "finalize": {"LLMPool"},
            }

            fanout = session.profiler.get_fanout_matrix()
            assert fanout["retrieve"]["VectorDB"] == 1.0
            assert fanout["generate"]["Cache"] == 1.0
            assert fanout["generate"]["LLMPool"] == 1.0
            assert fanout["finalize"]["LLMPool"] == 1.0

            counts = session.profiler.get_call_counts()
            assert counts["VectorDB"] == 1
            assert counts["Cache"] == 1
            assert counts["LLMPool"] == 2
            assert counts["ExternalService"] == 1

            external = session.profiler.get_external_call_counts()
            assert external["ExternalService"] == 1
            assert "ExternalService" not in deps


def test_profiling_fanout_avg_in_loops(shutdown_on_exit):
    @replica(n=2)
    class BatchProcessor:
        def process(self, item: str) -> str:
            return f"processed_{item}"

    @node
    async def process_batch(items: list) -> list:
        return [BatchProcessor.get().process(item) for item in items]

    BatchProcessor.deploy()

    with shutdown_on_exit(BatchProcessor):
        with profiling_enabled() as session:
            result = run(
                process_batch,
                ["a", "b", "c"],
                context_factory=session.context_factory,
            )
            assert len(result.data) == 3

            result = run(
                process_batch,
                ["1", "2", "3", "4", "5"],
                context_factory=session.context_factory,
            )
            assert len(result.data) == 5

            fanout = session.profiler.get_fanout_matrix()
            assert fanout["process_batch"]["BatchProcessor"] == 4.0

            counts = session.profiler.get_call_counts()
            assert counts["BatchProcessor"] == 8


def test_profiler_summary_and_reset_with_nested_nodes(shutdown_on_exit):
    @replica
    class ServiceA:
        def work_a(self) -> str:
            return "a"

    @replica
    class ServiceB:
        def work_b(self) -> str:
            return "b"

    @node
    async def node_a(_) -> str:
        return ServiceA.get().work_a()

    context_factory = None

    @node
    async def node_b(input_val: str) -> str:
        intermediate = await node_a(Context(input_val, context_factory=context_factory))
        return intermediate + ServiceB.get().work_b()

    ServiceA.deploy()
    ServiceB.deploy()

    with shutdown_on_exit(ServiceA, ServiceB):
        with profiling_enabled() as session:
            context_factory = session.context_factory
            result = run(node_b, "x", context_factory=session.context_factory)
            assert result.data == "ab"

            summary = session.profiler.summary()
            assert "dependency_graph" in summary
            assert "fanout_matrix" in summary
            assert "service_stats" in summary
            assert "call_counts" in summary
            assert "external_calls" in summary
            assert "node_executions" in summary
            assert "elapsed_seconds" in summary

            assert summary["dependency_graph"] == {
                "node_a": {"ServiceA"},
                "node_b": {"ServiceB"},
            }
            assert summary["fanout_matrix"] == {
                "node_a": {"ServiceA": 1.0},
                "node_b": {"ServiceB": 1.0},
            }
            assert summary["call_counts"] == {"ServiceA": 1, "ServiceB": 1}
            assert summary["node_executions"] == {"node_a": 1, "node_b": 1}
            assert summary["elapsed_seconds"] > 0

            session.profiler.reset()
            assert session.profiler.get_call_counts() == {}
            assert session.profiler.get_dependency_graph() == {}
            assert session.profiler.get_node_executions() == {}


def test_grpc_backend_end_to_end():
    """Test gRPC backend with server and client."""
    import socket
    import time

    pytest.importorskip("grpc", reason="gRPC backend requires the grpc package")
    from thinkagain.distributed.backend.grpc import ReplicaRegistry, serve

    # Define a replica class
    class Calculator:
        def __init__(self, multiplier: int = 1):
            self.multiplier = multiplier

        def multiply(self, x: int) -> int:
            return x * self.multiplier

    # Set up server with the class registered
    registry = ReplicaRegistry()
    registry.register(Calculator)

    try:
        server, port = serve(port=0, registry=registry)
    except RuntimeError as exc:
        pytest.skip(f"gRPC server unavailable in test environment: {exc}")

    # Wait for server to be ready with retry loop
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    else:
        server.stop(grace=0)
        pytest.fail("gRPC server failed to start within timeout")

    try:
        # Client side: use gRPC backend
        from thinkagain.distributed import init
        from thinkagain.distributed.replica import ReplicaSpec

        init(backend="grpc", address=f"localhost:{port}")

        spec = ReplicaSpec(cls=Calculator, n=2)

        # Deploy via gRPC
        spec.deploy_instances(multiplier=3)

        # Call via gRPC proxy
        proxy = spec.get_instance()
        result = proxy.multiply(10)
        assert result == 30

        # Shutdown
        spec.shutdown_instances()

    finally:
        server.stop(grace=0)

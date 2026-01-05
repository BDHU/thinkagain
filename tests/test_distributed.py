"""Slimmed down tests for distributed execution."""

import pytest

from thinkagain import Context, arun, chain, node, replica
from thinkagain.distributed import get_default_manager, runtime
from thinkagain.distributed.profiling import (
    disable_profiling,
    enable_profiling,
    get_profiler,
    is_profiling_enabled,
    node_context,
    profile,
)


@pytest.mark.asyncio
async def test_replica_registration_and_deploy():
    @replica(cpus=1)
    class Service:
        def __init__(self, value: int = 0):
            self.value = value

    spec = get_default_manager().get_spec("Service")
    assert spec.cls is Service.cls
    assert spec.cpus == 1
    assert spec.gpus == 0

    await Service.deploy(instances=2, value=5)
    # Get instances via backend
    inst1 = Service.get()
    inst2 = Service.get()
    assert inst1.value == 5
    assert inst2.value == 5

    await Service.shutdown()


@pytest.mark.asyncio
async def test_replica_round_robin():
    @replica(cpus=1)
    class Multiplier:
        def __init__(self, factor: int = 1):
            self.factor = factor
            self.call_count = 0

        def apply(self, x: int) -> int:
            self.call_count += 1
            return x * self.factor

    # Deploy two instances with same factor
    await Multiplier.deploy(instances=2, factor=2)

    # Round-robin should alternate between instances
    results = [Multiplier.get().apply(10) for _ in range(4)]
    assert results == [20, 20, 20, 20]  # Same factor, all 20


@pytest.mark.asyncio
async def test_replica_round_robin_distribution():
    """Test that round-robin actually distributes calls across different instances."""
    call_log = []

    @replica(cpus=1)
    class Logger:
        def __init__(self):
            pass

        def log(self):
            call_log.append(id(self))

    await Logger.deploy(instances=2)

    for _ in range(4):
        Logger.get().log()

    # Should see two distinct instances
    assert len(set(call_log)) == 2
    # Round-robin should alternate: [A, B, A, B]
    assert call_log[0] == call_log[2]
    assert call_log[1] == call_log[3]
    assert call_log[0] != call_log[1]


@pytest.mark.asyncio
async def test_pipeline_runs_with_replica():
    @replica(cpus=1)
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

    await Adder.deploy(delta=5)
    result = await arun(pipeline, 3)
    assert result.data == 9

    await Adder.shutdown()


@pytest.mark.asyncio
async def test_runtime_context_manager():
    @replica(cpus=1)
    class Service:
        def ping(self) -> bool:
            return True

    @node
    async def call_service(_) -> bool:
        return Service.get().ping()

    pipeline = chain(call_service)

    with runtime():
        await Service.deploy(instances=2)
        result = await arun(pipeline, None)
        assert result.data is True


@pytest.mark.asyncio
async def test_local_init_customizes_initialization():
    """Test that __local_init__ is called to customize instance creation."""
    init_calls = []

    @replica(cpus=1)
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

    await CustomInit.deploy(instances=2, base_value=10)

    # __local_init__ should have been called twice (instances=2)
    assert len(init_calls) == 2
    assert init_calls == [10, 10]

    # Instances should have the adjusted value
    inst1 = CustomInit.get()
    inst2 = CustomInit.get()
    assert inst1.value == 20
    assert inst2.value == 20


@pytest.mark.asyncio
async def test_local_init_fallback_to_regular_init():
    """Test that classes without __local_init__ use regular __init__."""

    @replica(cpus=1)
    class RegularInit:
        def __init__(self, value: int):
            self.value = value

    await RegularInit.deploy(value=5)

    inst1 = RegularInit.get()
    inst2 = RegularInit.get()
    assert inst1.value == 5
    assert inst2.value == 5


@pytest.mark.asyncio
async def test_profiling_toggle_and_context_manager():
    assert not is_profiling_enabled()
    assert get_profiler() is None

    profiler = enable_profiling()
    assert is_profiling_enabled()
    assert get_profiler() is profiler

    disable_profiling()
    assert not is_profiling_enabled()
    assert get_profiler() is None

    with profile() as profiler:
        assert is_profiling_enabled()
        assert get_profiler() is profiler

    assert not is_profiling_enabled()


@pytest.mark.asyncio
async def test_profiling_dependency_fanout_counts_external_calls(shutdown_on_exit):
    @replica(cpus=1)
    class VectorDB:
        def search(self, query: str) -> list:
            return [f"result_{query}"]

    @replica(cpus=1)
    class LLMPool:
        def generate(self, context: str) -> str:
            return f"response_{context}"

    @replica(cpus=1)
    class Cache:
        def lookup(self, key: str) -> str | None:
            return None

    @replica(cpus=1)
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

    await VectorDB.deploy(instances=2)
    await LLMPool.deploy(instances=2)
    await Cache.deploy(instances=2)
    await ExternalService.deploy(instances=2)

    with shutdown_on_exit(VectorDB, LLMPool, Cache, ExternalService):
        with profile() as profiler:
            result = await arun(
                pipeline,
                "test_query",
                context_factory=node_context,
            )
            assert "finalize_response" in result.data

            ExternalService.get().ping()

            deps = profiler.get_dependency_graph()
            assert deps == {
                "retrieve": {"VectorDB"},
                "generate": {"Cache", "LLMPool"},
                "finalize": {"LLMPool"},
            }

            fanout = profiler.get_fanout_matrix()
            assert fanout["retrieve"]["VectorDB"] == 1.0
            assert fanout["generate"]["Cache"] == 1.0
            assert fanout["generate"]["LLMPool"] == 1.0
            assert fanout["finalize"]["LLMPool"] == 1.0

            counts = profiler.get_call_counts()
            assert counts["VectorDB"] == 1
            assert counts["Cache"] == 1
            assert counts["LLMPool"] == 2
            assert counts["ExternalService"] == 1

            external = profiler.get_external_call_counts()
            assert external["ExternalService"] == 1
            assert "ExternalService" not in deps


@pytest.mark.asyncio
async def test_profiling_fanout_avg_in_loops(shutdown_on_exit):
    @replica(cpus=1)
    class BatchProcessor:
        def process(self, item: str) -> str:
            return f"processed_{item}"

    @node
    async def process_batch(items: list) -> list:
        return [BatchProcessor.get().process(item) for item in items]

    await BatchProcessor.deploy(instances=2)

    with shutdown_on_exit(BatchProcessor):
        with profile() as profiler:
            result = await arun(
                process_batch,
                ["a", "b", "c"],
                context_factory=node_context,
            )
            assert len(result.data) == 3

            result = await arun(
                process_batch,
                ["1", "2", "3", "4", "5"],
                context_factory=node_context,
            )
            assert len(result.data) == 5

            fanout = profiler.get_fanout_matrix()
            assert fanout["process_batch"]["BatchProcessor"] == 4.0

            counts = profiler.get_call_counts()
            assert counts["BatchProcessor"] == 8


@pytest.mark.asyncio
async def test_profiler_summary_and_reset_with_nested_nodes(shutdown_on_exit):
    @replica(cpus=1)
    class ServiceA:
        def work_a(self) -> str:
            return "a"

    @replica(cpus=1)
    class ServiceB:
        def work_b(self) -> str:
            return "b"

    @node
    async def node_a(_) -> str:
        return ServiceA.get().work_a()

    @node
    async def node_b(input_val: str) -> str:
        intermediate = await node_a(Context(input_val, context_factory=node_context))
        return intermediate + ServiceB.get().work_b()

    await ServiceA.deploy(instances=2)
    await ServiceB.deploy(instances=2)

    with shutdown_on_exit(ServiceA, ServiceB):
        with profile() as profiler:
            result = await arun(node_b, "x", context_factory=node_context)
            assert result.data == "ab"

            summary = profiler.summary()
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

            profiler.reset()
            assert profiler.get_call_counts() == {}
            assert profiler.get_dependency_graph() == {}
            assert profiler.get_node_executions() == {}


@pytest.mark.asyncio
async def test_multinode_incremental_scaling():
    """Test multi-node backend with incremental scaling.

    Demonstrates:
    - @replica(cpus=1) DemoClass
    - deploy(instances=2) spawns 2 servers
    - deploy(instances=3) spawns 1 more (scale up)
    - deploy(instances=1) shuts down 2 (scale down)
    """
    pytest.importorskip("grpc", reason="gRPC backend requires the grpc package")
    pytest.importorskip("cloudpickle", reason="Multi-node backend requires cloudpickle")

    from thinkagain.distributed.backend.multinode_grpc import MultiNodeGrpcBackend
    from thinkagain.distributed.replica import ReplicaSpec

    class DemoClass:
        def __init__(self, prefix: str = "demo"):
            self.prefix = prefix
            self.call_count = 0

        def process(self, data: str) -> str:
            self.call_count += 1
            return f"{self.prefix}:{data}:count={self.call_count}"

    backend = MultiNodeGrpcBackend()
    spec = ReplicaSpec(cls=DemoClass, cpus=1)

    try:
        # Initial deployment: 2 instances
        await backend.deploy(spec, instances=2, prefix="node")
        assert len(backend._servers["DemoClass"]) == 2

        # Test round-robin
        p1 = backend.get_instance(spec)
        r1 = await p1.process("a")
        assert r1 == "node:a:count=1"  # Server 1

        p2 = backend.get_instance(spec)
        r2 = await p2.process("b")
        assert r2 == "node:b:count=1"  # Server 2

        p3 = backend.get_instance(spec)
        r3 = await p3.process("c")
        assert r3 == "node:c:count=2"  # Server 1 again

        # Scale up to 3
        await backend.deploy(spec, instances=3, prefix="node")
        assert len(backend._servers["DemoClass"]) == 3

        # New server should be in rotation
        p4 = backend.get_instance(spec)
        r4 = await p4.process("d")
        assert r4 == "node:d:count=2"  # Server 2

        p5 = backend.get_instance(spec)
        r5 = await p5.process("e")
        assert r5 == "node:e:count=1"  # Server 3 (new)

        # Scale down to 1
        await backend.deploy(spec, instances=1, prefix="node")
        assert len(backend._servers["DemoClass"]) == 1

        # Only one server left
        p6 = backend.get_instance(spec)
        r6 = await p6.process("f")
        assert r6 == "node:f:count=3"  # Server 1 (the one that remained)

    finally:
        await backend.close()

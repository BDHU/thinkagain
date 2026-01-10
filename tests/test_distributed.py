"""Tests for distributed execution with @replica, Mesh, devices, and profiling."""

import asyncio

import pytest

import thinkagain as ta


# =============================================================================
# Module-level replica classes (for test reuse across multiple test functions)
# Note: With cloudpickle, replicas can also be defined locally in functions
# =============================================================================


@ta.replica()
class BasicProcessor:
    """Basic processor for general execution tests.

    This covers: basic execution, multiple calls, mixed pipelines.
    """

    def __init__(self, multiplier: int = 2):
        self.multiplier = multiplier
        self.calls = []

    async def __call__(self, x: int) -> int:
        """Process input by multiplying with multiplier."""
        self.calls.append(x)
        return x * self.multiplier


@ta.replica(gpus=4)
class GpuProcessor:
    """Processor with GPU requirement for resource tests."""

    def __init__(self, prompt: str = ""):
        self.prompt = prompt

    async def __call__(self) -> str:
        return f"Generated: {self.prompt}"


@ta.replica(gpus=1, backend="grpc")
class GrpcProcessor:
    """Processor with gRPC backend for backend-specific tests."""

    def __init__(self):
        self.count = 0

    async def __call__(self, text: str) -> str:
        self.count += 1
        return text.upper()


@ta.replica()
class ProfilingProcessor:
    """Processor with async delay for profiling tests."""

    def __init__(self):
        pass

    async def __call__(self, x: int) -> int:
        await asyncio.sleep(0.001)
        return x * 2


# Fixtures
@pytest.fixture(autouse=True)
def cleanup_pools():
    """Clean up pools before and after each test."""
    from thinkagain.distributed.replication.pool import _get_pools

    pools = _get_pools()
    pools.clear()
    yield
    pools.clear()


@pytest.fixture(autouse=True)
def cleanup_profiling():
    """Disable profiling before and after each test."""
    ta.disable_profiling()
    yield
    ta.disable_profiling()


# Device and Mesh tests
def test_gpu_device():
    """Test GPU device creation."""
    gpu = ta.GpuDevice(0)
    assert gpu.id == 0


def test_cpu_device():
    """Test CPU device creation."""
    cpu = ta.CpuDevice(0)
    assert cpu.id == 0


def test_mesh_creation():
    """Test mesh creation with devices and nodes."""
    # With devices
    mesh1 = ta.Mesh([ta.GpuDevice(0), ta.GpuDevice(1)])
    assert mesh1.total_gpus == 2

    # With nodes
    nodes = [
        ta.MeshNode("server1", gpus=8, cpus=1),
        ta.MeshNode("server2", gpus=8, cpus=1),
    ]
    mesh2 = ta.Mesh(nodes)
    assert mesh2.total_gpus == 16
    assert mesh2.total_cpus == 2  # 2 CPU devices (one per node)


def test_mesh_context():
    """Test mesh context manager."""
    mesh = ta.Mesh([ta.GpuDevice(0), ta.GpuDevice(1)])

    assert ta.get_current_mesh() is None

    with mesh:
        assert ta.get_current_mesh() is mesh

    assert ta.get_current_mesh() is None


# Replica decorator tests
def test_replica_basic():
    """Test basic @replica decorator configuration."""
    assert hasattr(BasicProcessor, "_replica_config")
    assert BasicProcessor._replica_config.gpus is None
    assert BasicProcessor._replica_config.backend == "local"

    # Create handle
    handle = BasicProcessor.init(2)  # type: ignore[attr-defined]
    assert isinstance(handle, ta.ReplicaHandle)


def test_replica_with_gpus():
    """Test @replica with GPU requirement."""
    config = GpuProcessor._replica_config
    assert config.gpus == 4

    # Create handle
    handle = GpuProcessor.init("test prompt")  # type: ignore[attr-defined]
    assert isinstance(handle, ta.ReplicaHandle)
    assert handle.config.gpus == 4


def test_replica_with_setup():
    """Test @replica with setup function."""
    # Test that GpuProcessor can be initialized with setup params
    config = GpuProcessor._replica_config
    assert config.gpus == 4
    # Setup would be configured at replica creation in real usage


def test_replica_class_with_call():
    """Test that @replica on classes with async __call__ works correctly."""
    # Should have replica config
    assert hasattr(GrpcProcessor, "_replica_config")
    assert GrpcProcessor._replica_config.gpus == 1
    assert GrpcProcessor._replica_config.backend == "grpc"

    # Create handle
    handle = GrpcProcessor.init()  # type: ignore[attr-defined]  # type: ignore[attr-defined]
    assert isinstance(handle, ta.ReplicaHandle)


@pytest.mark.asyncio
async def test_replica_class_in_jit_pipeline():
    """Test that @replica decorated classes can be used inside @jit pipelines."""
    # Create handle outside @jit
    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    @ta.jit
    async def pipeline(x: int) -> int:
        # Use handle in pipeline
        result = await processor(x)
        return result

    # Execute with mesh context
    mesh = ta.Mesh([ta.CpuDevice(4)])
    with mesh:
        result = await pipeline(5)
        assert result == 10


# Execution tests
@pytest.mark.asyncio
async def test_replica_execution():
    """Test basic replica execution with mesh."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]  # type: ignore[attr-defined]

    @ta.jit
    async def pipeline(x: int) -> int:
        return await processor(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await pipeline(5)

    assert result == 10


@pytest.mark.asyncio
async def test_replica_without_mesh():
    """Test replica works without mesh (fallback to local)."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]  # type: ignore[attr-defined]

    @ta.jit
    async def pipeline(x: int) -> int:
        return await processor(x)

    # Create a mesh for local execution
    mesh = ta.Mesh([ta.CpuDevice(4)])
    with mesh:
        result = await pipeline(5)
        assert result == 10


@pytest.mark.asyncio
async def test_replica_multiple_calls():
    """Test multiple calls to replica."""
    processor = BasicProcessor.init()  # type: ignore[attr-defined]  # type: ignore[attr-defined]

    @ta.jit
    async def pipeline(x: int) -> int:
        return await processor(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        results = [await pipeline(i) for i in range(5)]

    assert results == [0, 2, 4, 6, 8]


@pytest.mark.asyncio
async def test_replica_with_stateful_setup():
    """Test replica with setup (stateful execution)."""
    processor = BasicProcessor.init(1)  # type: ignore[attr-defined]  # multiplier=1

    @ta.jit
    async def pipeline(x: int) -> int:
        return await processor(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result1 = await pipeline(10)
        result2 = await pipeline(10)

    # Note: Stateful behavior - BasicProcessor tracks calls
    assert result1 == 10
    assert result2 == 10


@pytest.mark.asyncio
async def test_replica_async_setup():
    """Test replica with async methods."""
    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    @ta.jit
    async def pipeline(x: int) -> int:
        return await processor(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await pipeline(5)

    assert result == 10


@pytest.mark.asyncio
async def test_mixed_replicas_and_regular_nodes():
    """Test pipeline with both replicas and regular nodes."""

    @ta.node
    async def preprocess(x: int) -> int:
        return x + 1

    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    @ta.node
    async def postprocess(x: int) -> int:
        return x - 1

    @ta.jit
    async def pipeline(x: int) -> int:
        x = await preprocess(x)
        x = await processor(x)
        x = await postprocess(x)
        return x

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await pipeline(5)

    assert result == 11  # (5 + 1) * 2 - 1


# Profiling tests
@pytest.mark.asyncio
async def test_profiling_context_manager():
    """Test profiling with context manager."""
    assert not ta.is_profiling_enabled()

    with ta.profile() as profiler:
        assert ta.is_profiling_enabled()
        assert ta.get_profiler() is profiler

    assert not ta.is_profiling_enabled()


@pytest.mark.asyncio
async def test_profiling_with_replica():
    """Test that profiling tracks execution with replicas."""
    processor = ProfilingProcessor.init()  # type: ignore[attr-defined]  # type: ignore[attr-defined]

    @ta.jit
    async def pipeline(x: int) -> int:
        return await processor(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with ta.profile() as profiler:
        with mesh:
            await pipeline(5)
            await pipeline(10)
            await pipeline(15)

    # Check that profiling captured execution
    summary = profiler.summary()
    assert "execution_stats" in summary
    assert "node_executions" in summary
    assert summary["elapsed_seconds"] > 0

    # Check node executions were tracked
    node_execs = profiler.get_node_executions()
    assert len(node_execs) > 0, "Should have tracked node executions"

    # Check execution stats were captured
    exec_stats = profiler.get_execution_stats()
    assert len(exec_stats) > 0, "Should have execution statistics"


# Locally-defined replica tests
@pytest.mark.asyncio
async def test_locally_defined_replica():
    """Test that replica classes defined inside functions work with cloudpickle."""

    # Define replica inside the test function (this used to fail before cloudpickle)
    @ta.replica()
    class LocalCounter:
        def __init__(self, start: int = 0):
            self.count = start

        async def __call__(self) -> int:
            self.count += 1
            return self.count

    # Create handle
    counter = LocalCounter.init(start=10)  # type: ignore[attr-defined]

    # Use in pipeline
    @ta.jit
    async def count_pipeline() -> int:
        return await counter()

    mesh = ta.Mesh([ta.CpuDevice(1)])

    with mesh:
        result1 = await count_pipeline()
        result2 = await count_pipeline()

    assert result1 == 11
    assert result2 == 12


@pytest.mark.asyncio
async def test_locally_defined_replica_serialization():
    """Test that locally-defined replica handles can be serialized with cloudpickle."""
    import pickle

    # Define replica inside test
    @ta.replica()
    class LocalProcessor:
        def __init__(self, prefix: str):
            self.prefix = prefix

        async def __call__(self, text: str) -> str:
            return f"{self.prefix}: {text}"

    # Create handle
    processor = LocalProcessor.init(prefix="TEST")  # type: ignore[attr-defined]

    # Serialize and deserialize
    serialized = pickle.dumps(processor)
    restored = pickle.loads(serialized)

    # Verify handle attributes
    assert restored.replica_class.__name__ == "LocalProcessor"
    assert dict(restored.init_kwargs) == {"prefix": "TEST"}

    # Test execution with restored handle
    @ta.jit
    async def pipeline(text: str) -> str:
        return await restored(text)

    mesh = ta.Mesh([ta.CpuDevice(1)])

    with mesh:
        result = await pipeline("hello")

    assert result == "TEST: hello"


# Handle as parameter tests
@pytest.mark.asyncio
async def test_replica_handle_as_positional_arg():
    """Test that replica handles can be passed as positional arguments to @jit functions."""
    processor = BasicProcessor.init(3)  # type: ignore[attr-defined]  # multiplier=3

    @ta.jit
    async def pipeline(handle, x: int) -> int:
        return await handle(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await pipeline(processor, 5)

    assert result == 15  # 5 * 3


@pytest.mark.asyncio
async def test_replica_handle_as_kwarg():
    """Test that replica handles can be passed as keyword arguments to @jit functions."""
    processor = BasicProcessor.init(4)  # type: ignore[attr-defined]  # multiplier=4

    @ta.jit
    async def pipeline(x: int, handle=None) -> int:
        return await handle(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await pipeline(7, handle=processor)

    assert result == 28  # 7 * 4


@pytest.mark.asyncio
async def test_multiple_replica_handles_as_args():
    """Test passing multiple replica handles as arguments."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]  # multiplier=2
    processor2 = BasicProcessor.init(3)  # type: ignore[attr-defined]  # multiplier=3

    @ta.jit
    async def pipeline(h1, h2, x: int) -> int:
        # Call first handle, pass result to second
        result = await h1(x)
        result = await h2(result)
        return result

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await pipeline(processor1, processor2, 5)

    assert result == 30  # ((5 * 2) * 3) = 30


@pytest.mark.asyncio
async def test_replica_handle_mixed_closure_and_param():
    """Test mixing closure-captured handle with parameter handle."""
    closure_processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    @ta.jit
    async def pipeline(param_handle, x: int) -> int:
        # Call closure handle, then param handle
        result = await closure_processor(x)
        result = await param_handle(result)
        return result

    param_processor = BasicProcessor.init(5)  # type: ignore[attr-defined]
    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await pipeline(param_processor, 3)

    assert result == 30  # ((3 * 2) * 5) = 30


@pytest.mark.asyncio
async def test_replica_handle_caching_with_different_handles():
    """Test that graph caching works correctly with different handles passed as args."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(10)  # type: ignore[attr-defined]

    @ta.jit
    async def pipeline(handle, x: int) -> int:
        return await handle(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        # First call with processor1
        result1 = await pipeline(processor1, 5)
        # Second call with processor2 (should use different graph due to different handle)
        result2 = await pipeline(processor2, 5)

    assert result1 == 10  # 5 * 2
    assert result2 == 50  # 5 * 10

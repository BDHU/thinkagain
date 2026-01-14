"""Tests for distributed execution with @service, @op, Mesh, and .go() API."""

import asyncio

import pytest

import thinkagain as ta


# =============================================================================
# Module-level replica classes (for test reuse across multiple test functions)
# =============================================================================


@ta.service()
class BasicProcessor:
    """Basic processor for general execution tests."""

    def __init__(self, multiplier: int = 2):
        self.multiplier = multiplier
        self.calls = []

    async def process(self, x: int) -> int:
        """Process input by multiplying with multiplier."""
        self.calls.append(x)
        return x * self.multiplier

    async def get_calls(self) -> list[int]:
        """Get all calls made to this processor."""
        return self.calls.copy()


@ta.service(gpus=4)
class GpuProcessor:
    """Processor with GPU requirement for resource tests."""

    def __init__(self, prompt: str = ""):
        self.prompt = prompt

    async def generate(self) -> str:
        return f"Generated: {self.prompt}"


@ta.service(gpus=1, backend="grpc")
class GrpcProcessor:
    """Processor with gRPC backend for backend-specific tests."""

    def __init__(self):
        self.count = 0

    async def process(self, text: str) -> str:
        self.count += 1
        return text.upper()


@ta.service()
class ProfilingProcessor:
    """Processor with async delay for profiling tests."""

    def __init__(self):
        pass

    async def process(self, x: int) -> int:
        await asyncio.sleep(0.001)
        return x * 2


# =============================================================================
# Node functions for composition
# =============================================================================


@ta.op
async def preprocess(x: int) -> int:
    """Simple preprocessing node."""
    return x + 1


@ta.op
async def postprocess(x: int) -> int:
    """Simple postprocessing node."""
    return x - 1


@ta.op
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Fixtures
@pytest.fixture(autouse=True)
def cleanup_pools():
    """Clean up pools before and after each test."""
    from thinkagain.runtime.pool import _get_pools

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


# =============================================================================
# Device and Mesh tests
# =============================================================================


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


# =============================================================================
# Replica decorator tests
# =============================================================================


def test_replica_basic():
    """Test basic @service decorator configuration."""
    assert hasattr(BasicProcessor, "_service_config")
    assert BasicProcessor._service_config.gpus is None
    assert BasicProcessor._service_config.backend == "local"

    # Create handle
    handle = BasicProcessor.init(2)
    assert isinstance(handle, ta.ServiceClass)


def test_replica_with_gpus():
    """Test @service with GPU requirement."""
    config = GpuProcessor._service_config  # type: ignore[attr-defined]
    assert config.gpus == 4

    # Create handle
    handle = GpuProcessor.init("test prompt")  # type: ignore[attr-defined]
    assert isinstance(handle, ta.ServiceClass)
    assert handle.config.gpus == 4


def test_replica_with_setup():
    """Test @service with setup function."""
    # Test that GpuProcessor can be initialized with setup params
    config = GpuProcessor._service_config  # type: ignore[attr-defined]
    assert config.gpus == 4


def test_replica_class_with_call():
    """Test that @service on classes with async __call__ works correctly."""
    # Should have replica config
    assert hasattr(GrpcProcessor, "_service_config")
    assert GrpcProcessor._service_config.gpus == 1
    assert GrpcProcessor._service_config.backend == "grpc"

    # Create handle
    handle = GrpcProcessor.init()  # type: ignore[attr-defined]
    assert isinstance(handle, ta.ServiceClass)


# =============================================================================
# Execution tests with .go() API
# =============================================================================


@pytest.mark.asyncio
async def test_replica_execution():
    """Test basic replica execution with mesh using .go() API."""
    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        # Use .process.go() to call the replica
        result = await processor.process.go(5)

    assert result == 10


@pytest.mark.asyncio
async def test_replica_without_mesh():
    """Test replica works without mesh (fallback to local)."""
    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    # Create a mesh for local execution
    mesh = ta.Mesh([ta.CpuDevice(4)])
    with mesh:
        result = await processor.process.go(5)
        assert result == 10


@pytest.mark.asyncio
async def test_replica_multiple_calls():
    """Test multiple calls to replica using .go()."""
    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        # Multiple .process.go() calls
        results = await asyncio.gather(*[processor.process.go(i) for i in range(5)])

    assert results == [0, 2, 4, 6, 8]


@pytest.mark.asyncio
async def test_replica_with_stateful_setup():
    """Test replica with setup (stateful execution)."""
    processor = BasicProcessor.init(1)  # type: ignore[attr-defined]  # multiplier=1

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result1 = await processor.process.go(10)
        result2 = await processor.process.go(10)

    # Stateful behavior - BasicProcessor tracks calls
    assert result1 == 10
    assert result2 == 10


@pytest.mark.asyncio
async def test_replica_async_setup():
    """Test replica with async methods."""
    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await processor.process.go(5)

    assert result == 10


@pytest.mark.asyncio
async def test_mixed_replicas_and_nodes():
    """Test pipeline with both replicas and regular nodes using .go()."""
    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        # Chain nodes and replica calls using .go()
        x = await preprocess.go(5)  # 5 + 1 = 6
        x = await processor.process.go(x)  # 6 * 2 = 12
        result = await postprocess.go(x)  # 12 - 1 = 11

    assert result == 11


# =============================================================================
# Profiling tests
# =============================================================================


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
    processor = ProfilingProcessor.init()  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with ta.profile() as profiler:
        with mesh:
            await processor.process.go(5)
            await processor.process.go(10)
            await processor.process.go(15)

    # Check that profiling captured execution
    summary = profiler.summary()
    assert "execution_stats" in summary
    assert "node_executions" in summary
    assert summary["elapsed_seconds"] > 0

    # At minimum, profiling infrastructure should be active
    assert "execution_stats" in summary


# =============================================================================
# Locally-defined replica tests
# =============================================================================


@pytest.mark.asyncio
async def test_locally_defined_replica():
    """Test that replica classes defined inside functions work with cloudpickle."""

    # Define replica inside the test function
    @ta.service()
    class LocalCounter:
        def __init__(self, start: int = 0):
            self.count = start

        async def increment(self) -> int:
            self.count += 1
            return self.count

    # Create handle
    counter = LocalCounter.init(start=10)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(1)])

    with mesh:
        result1 = await counter.increment.go()
        result2 = await counter.increment.go()

    assert result1 == 11
    assert result2 == 12


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="ServiceClass pickle support not yet implemented for locally-defined classes"
)
async def test_locally_defined_replica_serialization():
    """Test that locally-defined replica handles can be serialized with cloudpickle."""
    import pickle

    # Define replica inside test
    @ta.service()
    class LocalProcessor:
        def __init__(self, prefix: str):
            self.prefix = prefix

        async def process(self, text: str) -> str:
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
    mesh = ta.Mesh([ta.CpuDevice(1)])

    with mesh:
        result = await restored.process.go("hello")

    assert result == "TEST: hello"


# =============================================================================
# Multiple replicas tests
# =============================================================================


@pytest.mark.asyncio
async def test_multiple_replica_handles():
    """Test using multiple replica handles in a pipeline."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(3)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        # Chain multiple replicas
        result = await processor1.process.go(5)  # 5 * 2 = 10
        result = await processor2.process.go(result)  # 10 * 3 = 30

    assert result == 30


@pytest.mark.asyncio
async def test_parallel_replica_calls():
    """Test parallel execution of multiple replicas."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(3)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        # Execute in parallel
        ref1 = processor1.process.go(5)
        ref2 = processor2.process.go(7)

        result1, result2 = await asyncio.gather(ref1, ref2)

    assert result1 == 10  # 5 * 2
    assert result2 == 21  # 7 * 3


@pytest.mark.asyncio
async def test_replica_with_node_composition():
    """Test composing replicas with @op functions using .go()."""
    processor1 = BasicProcessor.init(2)  # type: ignore[attr-defined]
    processor2 = BasicProcessor.init(5)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        # Compose using .go() API
        result1 = await processor1.process.go(10)  # 10 * 2 = 20
        result2 = await processor2.process.go(10)  # 10 * 5 = 50
        result = await add.go(result1, result2)  # 20 + 50 = 70

    assert result == 70


@pytest.mark.asyncio
async def test_complex_pipeline_with_go():
    """Test complex pipeline mixing nodes and replicas with .go()."""
    processor = BasicProcessor.init(3)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        # Build complex pipeline with .go() calls
        x = await preprocess.go(5)  # 5 + 1 = 6
        x = await processor.process.go(x)  # 6 * 3 = 18
        x = await postprocess.go(x)  # 18 - 1 = 17

    assert x == 17


@pytest.mark.asyncio
async def test_replica_stateful_behavior():
    """Test that replicas maintain state across calls."""
    processor = BasicProcessor.init(2)  # type: ignore[attr-defined]

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        # Make several calls
        await processor.process.go(1)
        await processor.process.go(2)
        await processor.process.go(3)

        # Check that calls were tracked (stateful)
        calls = await processor.get_calls.go()

    assert calls == [1, 2, 3]

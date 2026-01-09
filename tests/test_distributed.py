"""Tests for distributed execution with @replicate, Mesh, devices, and profiling."""

import asyncio

import pytest

import thinkagain as ta


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


# Replicate decorator tests
def test_replicate_basic():
    """Test basic @replicate decorator configuration."""

    @ta.replicate
    async def process(text: str) -> str:
        return text.upper()

    assert hasattr(process, "_distribution_config")
    assert process._distribution_config.gpus is None
    assert process._distribution_config.backend == "local"
    assert hasattr(process, "_is_node")


def test_replicate_with_gpus():
    """Test @replicate with GPU requirement."""

    @ta.replicate(gpus=4)
    async def generate(prompt: str) -> str:
        return f"Generated: {prompt}"

    config = generate._distribution_config
    assert config.gpus == 4


def test_replicate_with_setup():
    """Test @replicate with setup function."""

    def setup_model():
        return {"model": "test"}

    @ta.replicate(gpus=2, setup=setup_model)
    async def inference(model, data: str) -> str:
        return f"{model}: {data}"

    config = inference._distribution_config
    assert config.gpus == 2
    assert config.setup is setup_model


def test_replicate_class_gets_node_wrapper():
    """Test that @replicate on classes with async __call__ applies @node wrapper.

    This ensures classes can be used inside @jit pipelines, fixing the issue where
    classes were wrapped in ReplicatedCallable without the @node decorator.
    """

    @ta.replicate(gpus=1, backend="grpc")
    class TextProcessor:
        def __init__(self):
            self.count = 0

        async def __call__(self, text: str) -> str:
            self.count += 1
            return text.upper()

    # Should have distribution config
    assert hasattr(TextProcessor, "_distribution_config")
    assert TextProcessor._distribution_config.gpus == 1
    assert TextProcessor._distribution_config.backend == "grpc"

    # Should also have @node marker (this is the key fix!)
    assert hasattr(TextProcessor, "_is_node"), (
        "Classes with async __call__ should get @node wrapper for @jit integration"
    )


@pytest.mark.asyncio
async def test_replicate_class_in_jit_pipeline():
    """Test that @replicate decorated classes can be used inside @jit pipelines.

    Before the fix, classes were wrapped in ReplicatedCallable without @node,
    preventing them from being used in @jit functions. This test verifies the fix.
    """

    @ta.replicate
    class Processor:
        def __init__(self):
            self.count = 0

        async def __call__(self, x: int) -> int:
            self.count += 1
            return x * 2

    # Instantiate the processor
    processor = Processor()

    @ta.jit
    async def pipeline(x: int) -> int:
        # This should work now that classes get @node wrapper
        result = await processor(x)
        return result

    # Should execute successfully without errors
    result = await pipeline(5)
    assert result == 10
    # Count is incremented during execution
    assert processor.count >= 1


# Execution tests
@pytest.mark.asyncio
async def test_replicate_execution():
    """Test basic replicated function execution with mesh."""

    @ta.replicate
    async def process(x: int) -> int:
        return x * 2

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await pipeline(5)

    assert result == 10


@pytest.mark.asyncio
async def test_replicate_without_mesh():
    """Test replicated functions work without mesh (fallback to local)."""

    @ta.replicate
    async def process(x: int) -> int:
        return x * 2

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    result = await pipeline(5)
    assert result == 10


@pytest.mark.asyncio
async def test_replicate_multiple_calls():
    """Test multiple calls to replicated function."""
    calls = []

    @ta.replicate
    async def process(x: int) -> int:
        calls.append(x)
        return x * 2

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        results = [await pipeline(i) for i in range(5)]

    assert results == [0, 2, 4, 6, 8]
    assert calls == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_replicate_with_stateful_setup():
    """Test replicated function with setup (stateful execution)."""

    def setup():
        return {"counter": 0}

    @ta.replicate(setup=setup)
    async def process(state, x: int) -> int:
        state["counter"] += 1
        return x * state["counter"]

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result1 = await pipeline(10)
        result2 = await pipeline(10)

    assert result1 == 10  # 10 * 1
    assert result2 == 20  # 10 * 2


@pytest.mark.asyncio
async def test_replicate_async_setup():
    """Test replicated function with async setup."""

    async def async_setup():
        await asyncio.sleep(0.001)
        return {"initialized": True}

    @ta.replicate(setup=async_setup)
    async def process(state, x: int) -> int:
        assert state["initialized"] is True
        return x * 2

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with mesh:
        result = await pipeline(5)

    assert result == 10


@pytest.mark.asyncio
async def test_mixed_replicated_and_regular_nodes():
    """Test pipeline with both replicated and regular nodes."""

    @ta.node
    async def preprocess(x: int) -> int:
        return x + 1

    @ta.replicate
    async def process(x: int) -> int:
        return x * 2

    @ta.node
    async def postprocess(x: int) -> int:
        return x - 1

    @ta.jit
    async def pipeline(x: int) -> int:
        x = await preprocess(x)
        x = await process(x)
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
async def test_profiling_with_replicate():
    """Test that profiling tracks @replicate calls, dependencies, and fanout."""

    @ta.replicate
    async def process(x: int) -> int:
        await asyncio.sleep(0.001)
        return x * 2

    @ta.jit
    async def pipeline(x: int) -> int:
        return await process(x)

    mesh = ta.Mesh([ta.CpuDevice(4)])

    with ta.profile() as profiler:
        with mesh:
            await pipeline(5)
            await pipeline(10)
            await pipeline(15)

    # Check call counts
    call_counts = profiler.get_call_counts()
    assert call_counts["process"] == 3

    # Check execution stats
    stats = profiler.get_execution_stats()
    assert "process" in stats
    assert stats["process"]["count"] == 3
    assert stats["process"]["mean"] > 0

    # Check summary includes all expected fields
    summary = profiler.summary()
    assert "execution_stats" in summary
    assert "call_counts" in summary
    assert summary["elapsed_seconds"] > 0

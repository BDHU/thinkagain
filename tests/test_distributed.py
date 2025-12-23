"""Slimmed down tests for distributed execution."""

import pytest

from thinkagain import node, replica, run
from thinkagain.distributed import runtime
from thinkagain.distributed import (
    clear_replica_registry,
    get_replica_spec,
    reset_backend,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure each test starts with an empty registry."""
    clear_replica_registry()
    reset_backend()
    yield
    clear_replica_registry()
    reset_backend()


def test_replica_registration_and_deploy():
    @replica(n=2)
    class Service:
        def __init__(self, value: int = 0):
            self.value = value

    spec = get_replica_spec("Service")
    assert spec.cls is Service
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

    def pipeline(ctx):
        ctx = increment(ctx)
        ctx = apply_replica(ctx)
        return ctx

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

    def pipeline(ctx):
        ctx = call_service(ctx)
        return ctx

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
